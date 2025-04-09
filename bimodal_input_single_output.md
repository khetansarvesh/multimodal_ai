# Large MultiModal Models (LMMs)






# $\color{cyan}{Text\ +\ 2D\ Input\}$
1. Representation Learning (Visual Language Models - VLMs) : Here we can broadly classify the methods into two categories
   - [Coordinated / Parallel Stream Representation](https://khetansarvesh.medium.com/parallel-stream-representation-learning-for-visual-language-models-vlms-3b9233f3f8c5) : This type of encoding leads to two separate text and image embeddings which can be later used independently for unimodal tasks. 
   - [Joint / Fusion / Single Stream Representation](https://khetansarvesh.medium.com/single-stream-representation-learning-for-visual-language-models-vlms-b9455b35216a) : This type of encoding leads to one single embedding for both text and image.

2. [Downstream Tasks](https://khetansarvesh.medium.com/downstream-tasks-using-vlms-57be1fadb618)


Some standard datasets for VLM are as follows : 
- [Microsoft Common Objects in Context (COCO)](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
- Visual Genome
- Conceptual Captions (CC)
- ADE20K
- [VQA](https://visualqa.org/) v2
- [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)
- [LAION](https://laion.ai/) - 400M and LAION-5B
- Localized Narratives
- WuDaoMM
- Wikipedia Image Text (WIT)








# $\color{cyan}{Text\ +\ Video\ Input\}$
1. Representation Learning :
   - [VideoBERT](https://arxiv.org/abs/1904.01766) (Google, April 2019)
   - ActBERT
     
2. Downstream Tasks :
   - Built In Video Search eg suppose in a movie of 2hr you want to search for a car exploding scene
   - Video Captioning
   - Video Retrieval i.e. given a txt prompt retrieve all the relevant videos (youtube algorithm)











# $\color{cyan}{Text\ +\ Audio\ Input\}$
1. Representation Learning (Audio Language Model) :
   - [GAMA](https://sreyan88.github.io/gamaaudio/)
   
2. Downstream Tasks








# $\color{cyan}{Text\ +\ Tabular\ Data\ Input\}$
1. Representation Learning
   - Use Sequence model like RNN / LSTM / …./ Transformers to encode text data 
   - ANN to encode tabular cross sectional data 
   - Then concatenate the outputs from last hidden layer of these networks and then again train the combined output with a FFNN
     
2. Downstream Tasks
   - Tabular QA Task









# $\color{cyan}{2D\ Image\ +\ Tabular\ Data\ Input\}$
1. Representation Learning
   - Use CNN to encode image data 
   - ANN to encode tabular cross sectional data 
   - Then concatenate the outputs from last hidden layer of these networks and then again train the combined output with a FFNN
     
2. Downstream Tasks










# $\color{cyan}{Video\ +\ Audio\ Input\}$
1. Representation Learning (Video Large Language Model or VLLM) : 
   - To handle video : combo of CNN (to convert image to numerical form) / vision transformers + RNN (to handle sequence of images information) / transformers
   - To handle audio :

2. Downstream Tasks :
   - Video Generation
  




# $\color{cyan}{Mask Prompts\ +\ Image\ Input\}$
[SAM1 by Meta](https://www.youtube.com/watch?v=eYhvJR4zFUM)


# $\color{cyan}{Mask Prompts\ +\ Video\ Input\}$
[SAM2 by Meta](https://www.youtube.com/watch?v=wMGb97EZkVU)
