# $\color{cyan}{1.\ Cross\ Modal\ System\}$

## [Natural Language Generation Tasks (X2Text) Task](https://github.com/khetansarvesh/multimodal_ai/tree/main/cross_modal)
- Image2Text i.e. Image Captioning
  - Think of Text2Text task eg Machine Translation and now instead of conditioning it on text we condition it on Image

- Audio2Text / Speech2Text eg audio transcription (also called Speech Recognition or Automatic Speech Recognition (ASR) System)
  - Think of Text2Text task eg Machine Translation and now instead of conditioning it on text we condition it on Audio. Hence we can use exactly what we saw in Image2Text just that now instead of conditioning it on input image embeddings we will condition it on input audio embeddings !!
  - Important Model : Whisper by OpenAI

- Video2Text i.e. Video Captioning
  - Think of Text2Text task eg Machine Translation and now instead of conditioning it on text we condition it on Video

## Text2X Task











# $\color{cyan}{2.\ Multimodal\ Inputs\ Single\ Output\ System\}$
Here the multimodal inputs can be of different types : 
- [Bimodal](https://github.com/khetansarvesh/multimodal_ai/blob/main/multimodal_input_single_output/bimodal_input_single_output.md) Input Single Output :-
  
- Trimodal Input Single Output :-
  - [MERIOT](https://arxiv.org/pdf/2201.02639) : Handles Image, Audio and Text
  - [VATT](https://arxiv.org/pdf/2104.11178) : Handles Video, Audio and Text
  - [ULIP](https://arxiv.org/abs/2212.05171) : Handles Text, Images and 3D Point Clouds
    
- General Input Single Output :-
  - [ImageBind](https://arxiv.org/pdf/2305.05665) : Capable of handling 6 different input modalities i.e. images, text, audio, depth, thermal, and IMU data.
  - [Meta-Transformer](https://kxgong.github.io/meta_transformer/) : Capable of handling text, images, audio, 3D point clouds, video, graphs, and tables (both cross sectional and time series)
  - [Macaw](https://arxiv.org/abs/2306.09093) : Capable of handling Image, Audio, Video, Text











# $\color{cyan}{3.\ Single\ Input\ Multimodal\ Output\ System\}$
These are systems that can generate multiple modalaties as output for instance if we ask ChatGPT to explain ’what is AI’ then an effective explanation might require graphs, equations, and even simple animations.

To generate multimodal outputs, a model would first need to generate a shared intermediate output. One key question is what the intermediate output would look like.

### <ins> Method 1 </ins> : 
- One option for intermediate output is text, which will then be used to generate/synthesize other actions.
- For example, [CM3](https://arxiv.org/abs/2201.07520) outputs HTML markup which can be compiled into web pages that contain not only text but also formattings, links, and images. GPT-4V generates Latex code, which can then be reconstructed as data tables.

### <ins> Method 2 </ins> : 
- Another option for intermediate output would be multimodal tokens.
- Each token will have a tag to denote whether it’s a text token or an image token. 
  - Image tokens will then be input into an image model like Diffusion to generate images. (or use already existing models to generate image)
  - Text tokens will then be input into a language model (or use already existing model to generate text)
- [Generating Images with Multimodal Language Models](https://arxiv.org/abs/2305.17216) is an awesome paper that shows how LMMs can generate and retrieve images together with generating texts.










# $\color{cyan}{4.\ Multimodal\ Input\ and\ Multimodal\ Outputs\ System\}$
These are systems that can generate both text and images.
- An example model is [Next-GPT](https://next-gpt.github.io/)
- These two CVPR sessions talk more about building such models
  a. [Video1](https://www.youtube.com/watch?v=pHBT3zXxQX8)
  b. [Video2](https://www.youtube.com/watch?v=mkI7EPD1vp8)

