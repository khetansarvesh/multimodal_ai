# ****** Natural Language Generation Tasks (x2Text) ******

### 1. Image2Text i.e. Image Captioning
- Think of Text2Text task eg Machine Translation and now instead of conditioning it on text we condition it on Image

### 2. Audio2Text / Speech2Text eg audio transcription (also called Speech Recognition or Automatic Speech Recognition (ASR) System)
- Think of Text2Text task eg Machine Translation and now instead of conditioning it on text we condition it on Audio. Hence we can use exactly what we saw in Image2Text just that now instead of conditioning it on input image embeddings we will condition it on input audio embeddings !!
- Important Model : Whisper by OpenAI

### 3. Video2Text i.e. Video Captioning
- Think of Text2Text task eg Machine Translation and now instead of conditioning it on text we condition it on Video

# ****** Text2X Tasks ******

### 1. Text2Image (TTI) Task
- Think of Image2Image Task and then condition it on Text.
- Now we have seen Image2Image task [here](https://levelup.gitconnected.com/image-data-augmentation-techniques-d9323f22153f) and we saw all the generative models to do this i.e. Autoencoders / GAN / DDPMs / DDIMs but here during inference you won't have a input image you will only have an input text hence while training you will have to use a model which convert noise to image for the image2image task and hence we can only use models like GANs and Diffusion for Text2Image Task !!
- Inpainting
- Outpainting
- Some famous Models are : Midjourney, Stable Diffusion, and Dall-E (these are all diffusion based models and hence the model weights can be downloaded using Hugging Face's diffusers library, example code available [here](https://github.com/khetansarvesh/multimodal_ai/blob/main/cross_modal/txt2img_hf.ipynb) )
