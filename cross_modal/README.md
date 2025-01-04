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

### 2. Text2Speech Task (TTS) (also called Speech Synthesis)
- Think of Speech2Speech Task and then condition it on texts, same as logic explained above we can here also only use models like GANs and Diffusion
- [AudioGen by Facebook](https://arxiv.org/pdf/2209.15352.pdf)
- [MusicGen by Facebook](https://arxiv.org/pdf/2306.05284.pdf)
- [Indic-TTS](https://ai4bharat.iitm.ac.in/indic-text-to-speech/) developed by AI4Bharat is capable of generating speech in multiple Indian Languagues. Demo available [here](https://models.ai4bharat.org/#/tts). It involves two different models :
  - An acoustic model, which is responsible for generating waveform from a given text
  - A vocoder model, which is responsible for synthesizing voice from the generated waveform. Currently, we use FastPitch and HiFi-GAN models.

  
### 3. Text2Video Task
- Think of Video2Video Task and then condition it on texts, same as logic explained above we can here also only use models like GANs and Diffusion
- You can learn a general framework [here](https://www.youtube.com/watch?v=0K56LA821ys)
- Some prominent models in this space are :
  1. [Stable Video Diffusion](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)
  2. [Lumiere By Google](https://www.youtube.com/watch?v=Pl8BET_K1mc)
  3. [Make A Video By Meta](https://makeavideo.studio/) => Here is a [Blog](https://medium.com/@lakshmibayanagari/metas-make-a-video-breakdown-8d8618c7b8e8) explaining this paper
  4. SORA by Open AI
  5. Veo2 by Google
