# simple text generation tutorials 
AndrejKarpathy_SimpleTextGen_pytorch.py - uses a bigram model trained on cross entropy to do text generation. Follow along with this tutorial 
https://www.youtube.com/watch?v=kCc8FmEb1nY&t=776s -- Will end up showing you how to do the full transformer in pytorch. current progress on this is at the bigram model

SimpleTexGen_tensorflow.py -- uses a transformer model implemented in tensorflow to do the same text generation.
transformer_parts.py -- has the custom tensorflow layres for the Simple_TextGen_tensorflow.py example. Model implments the transformer but could use some polishing on the loss funciton. 

# hugging face models and usage 
hugging face model card will have everything that the model is used for, how it was trained, etc. 
For example https://huggingface.co/FacebookAI/roberta-large-mnli
this one is a text classificaiton model -- built on BERT -- bidirectional encoder representations from transformers
Multi-genre Natural language inference (MNLI)
Trained on masked language modeling objective. 

-- Working in google colab -- it's all ipython note books
Experimenting with Hugging Face.ipynb -- ok this exmample loads up BERT 
Turns out any google account has access to google collab which comes with huging face transformers preinstalled
So build your apps there. 


