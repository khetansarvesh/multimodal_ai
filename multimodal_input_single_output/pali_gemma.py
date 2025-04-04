import math
from typing import Dict, List, Optional, Union, Tuple, Iterable
import fire

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn as nn


# self.vocab_size = vocab_size
# self.hidden_size = hidden_size
# self.intermediate_size = intermediate_size
# self.num_hidden_layers = num_hidden_layers
# self.num_attention_heads = num_attention_heads
# self.num_key_value_heads = num_key_value_heads
# self.max_position_embeddings = 8192
# self.head_dim = 256
# self.rms_norm_eps = 1e-6
# self.rope_theta = 10000.0
# self.attention_bias = False
# self.attention_dropout = 0.0
# self.pad_token_id = None


class VLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = -1
        self.padding_idx = None
        self.hidden_size = hidden_size # text_config.hidden_size

        img_size=224 
        in_channels=3 
        patch_size=16
        num_patches = (img_size // patch_size) * (img_size // patch_size) #num_patches = (img_size * img_size) // patch_size**2
        self.im_position_embedding = nn.Parameter(data=torch.randn(1, num_patches, 768),requires_grad=True)

    def vision_model(self, x):
        # 32, 1, 28, 28 -> 32, 1, 7*4, 7*4 -> 32, 1, 7, 7, 4, 4 -> 32, 7, 7, 4, 4, 1 -> 32, 7*7, 4*4*1 - > 32, num_patches, patch_dim
        x = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)', ph=4, pw=4)

        # Create patch embedding for all images in the batch
        x = nn.Sequential(nn.LayerNorm(1*4*4), nn.Linear(1*4*4, 768), nn.LayerNorm(768))(x)

        # Add position embedding to patch embedding
        x = self.im_position_embedding + x

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x =  nn.TransformerEncoder( encoder_layer = nn.TransformerEncoderLayer(d_model=768, # Hidden size
                                                                            nhead=2,
                                                                            dim_feedforward=3072, ## FFNN hidden size
                                                                            activation="gelu",
                                                                            batch_first=True,
                                                                            norm_first=True), # Create a single Transformer Encoder Layer
                                                        num_layers=2)(x) # Stack it N times

        x = nn.LayerNorm(768, eps=1e-6)(x)

        return x

    def language_model(self, attention_mask, position_ids, x, kv_cache):

        x = x * torch.tensor(self.hidden_size**0.5, dtype=x.dtype)

        x = GemmaDecoderLayer(config, 1)( x, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)
        x = GemmaDecoderLayer(config, 2)( x, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)
        x = GemmaDecoderLayer(config, 3)( x, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)
        x = GemmaDecoderLayer(config, 4)( x, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)

        logits = nn.Linear(self.hidden_size, self.vocab_size, bias=False)(x).float()

        return_data = {"logits": logits}

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data

    def _merge_input_ids_with_image_features( self, image_features, inputs_embeds, input_ids, attention_mask, kv_cache):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (2048**0.5)
    
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != 256000) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == 256000
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            # Since we are generating tokens, the query must be one single token
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask => For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward( self, x, pixel_values, attention_mask, kv_cache):

        # get image embeddings and project it to match text embeddings
        inputs_embeds = nn.Embedding(vocab_size, hidden_size, None)(x) #language model hidden size
        selected_image_feature = self.vision_model(pixel_values.to(inputs_embeds.dtype))
        image_features = nn.Linear(768, 2048, bias=True)(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        output = self.language_model(attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, kv_cache=kv_cache)

        return output
    


self.image_token_index = 256000
self.projection_dim = 2048
self.hidden_size = 2048
self.vision_config = None
self.is_encoder_decoder = False
self.pad_token_id = None

self.vision_config = SiglipVisionConfig(**vision_config)
self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
self.vocab_size = self.text_config.vocab_size #257152

self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2