# Large MultiModal Models (LMMs)






# $\color{cyan}{Text\ +\ 2D\ Input\}$
1. Representation Learning (Visual Language Models - VLMs) : Here we can broadly classify the methods into two categories
   - Coordinated / Parallel Stream Representation
   - Joint / Fusion / Single Stream Representation

2. [Downstream Tasks](https://khetansarvesh.medium.com/downstream-tasks-using-vlms-57be1fadb618)










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
