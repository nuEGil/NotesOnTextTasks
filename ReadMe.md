# Text features and knowldege graph construction 
Need to look into neo4j -- graph RAG tool for AI models. So this tool is inspired by that and this YouTube video https://www.youtube.com/watch?v=j2T4gvQAiaE&t=567s where the creator uses a graph based approach to studying friendships between characters in top mangas. Find words (people, proteins, genes) that occur in the same episode (window of text). This is relavent for LLMs because as users pass inputs to the model their context windows fill up. Graph based approaches are just one class that can be used to create data structures that might reduce the amount of information you need to pass to the context window to get good results. -- SO I figure I need to build some text processing tools that are not totally AI. Maybe part of the context over load leads to positive reinforcement cycles that are bad for people's health. 

Existing tools : neo4j, Letta, Anthropic Claude Memory management.  
https://www.letta.com/

1. FeatureCrafting.py -- initial pass at extracting nouns, sentences, words, etc. 
2. TapeReader.py -- Secondary pass that uses regex patterns to pull paragraphs, sentences, gets nouns, but has some hard filtereing steps to reduce the number of word pairs used for the graph construction. Checks for the occurence of word pairs in windows of words (n sentences). Currently gets end count of word pairs -- going to expand to get indices of increment for the word pairs -- then save the whole thing to a JSON. So you have word1, word2, list of indices. --> from there we can do text classifiers to look for more interesting relationships. 
3. Graph_making.py -- use pyvis to make an html graph of the csv file dumped from TapeReader.py.

Next step for this --> make a google collab to demonstrate functions 
Train a Text Classifier for finding more interesting + abstract relationships.
Make a GUI to bundle it all up -- Kivy here we come.  


# Simple text generation tutorials 
these scripts are for when we want to train something from scratch using pytorch and tensorflow. Modify layers, modify inputs. custom as much as we can -- while making use of torch layers and keras layers as much as possible

1. AndrejKarpathy_SimpleTextGen_pytorch.py
uses a bigram model trained on cross entropy to do text generation. Follow along with this tutorial 
https://www.youtube.com/watch?v=kCc8FmEb1nY&t=776s -- Will end up showing you how to do the full transformer in pytorch. current progress on this is at the bigram model

2. SimpleTexGen_tensorflow.py
uses a transformer model implemented in tensorflow to do the same text generation.
transformer_parts.py -- has the custom tensorflow layres for the Simple_TextGen_tensorflow.py example. Model implments the transformer but could use some polishing on the loss funciton. 

3. CardiacSignalsStuff.ipynb
This example uses the MIT-BIH arrhymia data set (copy on physionet) which contains ECG signals from multiple views along with data like patient age, sex, and medication lists. The data was taken in the 80s and digitized. In this example we load some of the data with the wfdb library, show the frequency spectrum and filter the signal. I also wrote a simple algorithm to extract the R-R intervals. I generate prompts given the patient meta data and the R-R intervals, and pass those prompts to gpt2 using the huggingface hosted gpt2 model. There's some interesting results here. The model definitely needs some finetuning, but it is a cool way to test out something of a full pipeline that involves: signal loading, cleaning, windowing, phyisological feature extraction, followed by prompting for text generation. The options here would be to fine tune gpt2 or to put some more work into the SimpleTextGen_tensorflow.py model. But for now this is enough. 



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
