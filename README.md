# 1. Cross Modal System
These are systems that take input and give output modalities which are of different types. More information on this available [here](https://github.com/khetansarvesh/multimodal_ai/tree/main/cross_modal)

# 2. Multimodal Inputs Single Output System 
These are systems that can take both text and images as inputs
- Such models are commonly know as Large MultiModal Models (LMMs) 


# 3. Single Input Multimodal Output System
These are systems that can generate multiple modalaties as output for instance if we ask ChatGPT to explain ’what is AI’ then an effective explanation might require graphs, equations, and even simple animations.

To generate multimodal outputs, a model would first need to generate a shared intermediate output. One key question is what the intermediate output would look like.

### 1. Method 1 : 
- One option for intermediate output is text, which will then be used to generate/synthesize other actions.
- For example, [CM3](https://arxiv.org/abs/2201.07520) outputs HTML markup which can be compiled into web pages that contain not only text but also formattings, links, and images. GPT-4V generates Latex code, which can then be reconstructed as data tables.

### 2. Method 2 : 
- Another option for intermediate output would be multimodal tokens.
- Each token will have a tag to denote whether it’s a text token or an image token. 
  - Image tokens will then be input into an image model like Diffusion to generate images. (or use already existing models to generate image)
  - Text tokens will then be input into a language model (or use already existing model to generate text)
- [Generating Images with Multimodal Language Models](https://arxiv.org/abs/2305.17216) is an awesome paper that shows how LMMs can generate and retrieve images together with generating texts.




# 4. Multimodal Input and Multimodal Outputs System
These are systems that can generate both text and images.
- An example model is [Next-GPT](https://next-gpt.github.io/)
- These two CVPR sessions talk more about building such models
  a. [Video1](https://www.youtube.com/watch?v=pHBT3zXxQX8)
  b. [Video2](https://www.youtube.com/watch?v=mkI7EPD1vp8)

