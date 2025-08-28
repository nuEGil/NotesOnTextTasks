# Simple text generation tutorials 
these scripts are for when we want to train something from scratch using pytorch and tensorflow. Modify layers, modify inputs. custom as much as we can -- while making use of torch layers and keras layers as much as possible

1. AndrejKarpathy_SimpleTextGen_pytorch.py
uses a bigram model trained on cross entropy to do text generation. Follow along with this tutorial 
https://www.youtube.com/watch?v=kCc8FmEb1nY&t=776s -- Will end up showing you how to do the full transformer in pytorch. current progress on this is at the bigram model

2. SimpleTexGen_tensorflow.py
uses a transformer model implemented in tensorflow to do the same text generation.
transformer_parts.py -- has the custom tensorflow layres for the Simple_TextGen_tensorflow.py example. Model implments the transformer but could use some polishing on the loss funciton. 

# Hugging face models and usage 
This stuff 
hugging face model card will have everything that the model is used for, how it was trained, etc. 
For example https://huggingface.co/FacebookAI/roberta-large-mnli
this one is a text classificaiton model -- built on BERT -- bidirectional encoder representations from transformers
Multi-genre Natural language inference (MNLI)
Trained on masked language modeling objective. 

## Google Colab
1. Experimenting with Hugging Face.ipynb 
ok this exmample loads up BERT Turns out any google account has access to google collab which comes with huging face transformers preinstalled
So build your apps there. 

## x-ray images 
1. patchmaker.py -- use this to extract patches from the main image that we can train on 
2. local_tf_ResNetTuner.py

# Todo list
1. knowledge graph implementation 

2. tf or pytorch - local model: Fine Tune a ResNet model on image patches of chest x-rays, and use that to do a segmentation 
    finetuning done --> implmement a contrastive learning algorithm next. 

3. tf - local model : Train a U-Net to segment out the chest x-ray. 

4. Google colab --> import the trained U-Net model to do the semgnetaiton , then use a multimodal text + vision model 

look into texts that are publicly available, that you might fine tune a model for chest anatomy knowledge.... likely would need 
the pairs to get this to actually work, but we have time now for a proof of conceptat least. 

may implement vision transformer- but resnet and U-Net have been pretty effective for my use cases so far -- see convnet for the 2020s paper. 
