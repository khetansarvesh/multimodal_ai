import math
from typing import Dict, List, Optional, Union, Tuple, Iterable
import fire

import numpy as np
from PIL import Image
from einops import rearrange

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn as nn

class VLM(nn.Module):

    def __init__(self, vocab_size = 30522, hidden_size = 768, img_size=224, patch_size=16, num_heads=8, num_layers=2):
        super(VLM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_dim = 3 * patch_size * patch_size  # Input channels (3) * patch size

    def vision_model(self, x):
        # 32, 1, 28, 28 -> 32, 1, 7*4, 7*4 -> 32, 1, 7, 7, 4, 4 -> 32, 7, 7, 4, 4, 1 -> 32, 7*7, 4*4*1 - > 32, num_patches, patch_dim
        x = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)', ph=4, pw=4)

        # Create patch embedding for all images in the batch
        x = nn.Sequential(nn.LayerNorm(1*4*4), nn.Linear(1*4*4, 768), nn.LayerNorm(768))(x)

        # Add position embedding to patch embedding
        x += nn.Parameter(data=torch.randn(1, self.num_patches, 768),requires_grad=True)

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

    def forward(self, 
                text_tokens, # these are text tokens like <223 98 738 67 03 ........ 890>
                im):

        '''Convert text tokens (scalars) to embeddings (vectors) '''
        text_features = nn.Embedding(self.vocab_size, 2048)(text_tokens)

        '''Process image features through the vision model'''
        image_features = self.vision_model(im)
        image_features = nn.Linear(768, 2048, bias=True)(image_features) # matching image embeddings with text embeddings

        '''Concatenate image and text features along the sequence dimension'''
        combined_features = torch.cat([image_features, text_features], dim=1)

        '''passing the concatenated tokens via transformer encoder'''
        combined_features = nn.TransformerEncoder(
                                    nn.TransformerEncoderLayer(
                                        d_model=2048,
                                        nhead=8,
                                        dim_feedforward=2048 * 4,
                                        activation="gelu",
                                        batch_first=True
                                    ),
                                    num_layers=4
                                )(combined_features)

        '''Compute logits for the combined features'''
        logits = nn.Linear(2048, self.vocab_size)(combined_features)

        return logits
