from AC_pythonexample import * # will already have json and time imported  
import torch 
from transformers import DistilBertTokenizer, DistilBertModel
'''
distilebert tokenization and classification example 
generic for interviews. 
'''
def load_model_and_tokenizer(name_ = 'distilbert-base-uncased'):
    tokenizer = DistilBertTokenizer.from_pretrained(name_)
    model = DistilBertModel.from_pretrained(name_, num_labels = 2)
    classifier = torch.nn.Linear(model.config.hidden_size, 2)
    return model, tokenizer, classifier

def parse_(text, model, tokenizer, classifier):
    
    inputs = tokenizer(text, return_tensors="pt") # tokenize the input
    # forward pass through the model 
    with torch.no_grad():
        outputs = model(**inputs) # pass all tokenized inputs to the model
        hidden_states = outputs.last_hidden_state # (batch_size, seq_len, hidden_dim)
        avg_pool = hidden_states.mean(dim=1)
    # pooled representation through classifier 
    logits = classifier(avg_pool)
    pred_class = torch.argmax(logits, dim = -1).item()
    
    print(f"\nText: '{text}'")
    print(f"Predicted class: {pred_class} (0 for negative, 1 for positive)")

if __name__ == '__main__':

    input_file = "/mnt/f/data/ebooks_public_domain/crime and punishment.txt"
    # get full text
    text = GetText(input_file)
    # get sentences
    sentence_starts, sentence_ends, sentences = GetSentences0(text, offset = 0)
    model, tokenizer, classifier = load_model_and_tokenizer()
    
    for i in range(10):
        print('---')
        parse_(sentences[100+i], model, tokenizer, classifier)

