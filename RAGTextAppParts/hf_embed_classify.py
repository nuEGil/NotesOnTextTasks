from AC_pythonexample import * # will already have json and time imported  
import torch 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
'''
distilebert tokenization and classification example 
generic for interviews. 
'''

def load_model_and_tokenizer(name_ = 'distilbert-base-uncased'):
    tokenizer = DistilBertTokenizer.from_pretrained(name_)
    model = DistilBertForSequenceClassification.from_pretrained(name_, num_labels = 2)
    return model, tokenizer

def parse_(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt") # tokenize the input
    outputs = model(**inputs) # pass all tokenized inputs to the model
    predicted_class = outputs.logits.argmax().item()
    
    print(f"\nText: '{text}'")
    print(f"Predicted class: {predicted_class} (0 for negative, 1 for positive)")

if __name__ == '__main__':

    input_file = "/mnt/f/data/ebooks_public_domain/crime and punishment.txt"
    # get full text
    text = GetText(input_file)
    # get sentences
    sentence_starts, sentence_ends, sentences = GetSentences0(text, offset = 0)
    model,tokenizer = load_model_and_tokenizer()
    
    for i in range(10):
        print('---')
        parse_(sentences[100+i], model, tokenizer)

