import os 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.nn import functional as F


def LoadTextChars(fpath):
    '''Nothing fancy text reader. we want to load text 
    so that it is already structured just get character sequences
    hence f.read() and not f.readlines()
    '''
    with open(fpath,'r') as f:
        text = f.read()
    return text

def CharEncodingTool(chars_):
    '''Easy tokenizer -- tensorflow has some other tokenizers we can use though
    some are subword tokenizers --> encode by syllables - in the case of genetic sequences you would 
    want something for Codons and protein sequences. For SMILEs you want arbitrary chunks to figure out chem structure
    
    Character level gives long sequences though
    '''
    
    string_to_int = {ch:i for i,ch in enumerate(chars_)}
    int_to_string = {i:ch for i,ch in enumerate(chars_)}
    encode  = lambda s: [string_to_int[c] for c in s] # encoder take a string output a list of ints
    decode  = lambda l : ''.join(int_to_string[i] for i in l) # decoder -- convert integers to output string
    return encode, decode

# Basic model for testing purposes

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token froma  lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        return logits

if __name__ == '__main__':
    '''Doing this at the character level sets this code up for use with gene sequencing and SMILEs    
    You could do it at the word level though if your application is to learn something about words. 
    Might make a letter, word, sentence, paragaph exploration tool at somepoint.
    '''
 
    # read a book
    fpath = '/mnt/f/ebooks_public_domain/Poisonous snakes of texas and first aid treatment of their bites.txt'
    text = LoadTextChars(fpath)

    # get set of unique characters - in a list object
    num_chars = len(text)
    unique_chars = sorted(list(set(text)))
    vocab_size = len(unique_chars)

    # print relavent info 
    print('number of characters', num_chars)
    print('vocab size: ', vocab_size)
    print('unique chars', unique_chars)

    print(unique_chars)
    # print out the header -- arbitrary point in the text. 
    print(text[1000:2000])
    
    encode, decode = CharEncodingTool(unique_chars)
    print('Example of encoding text')
    input_string = 'There is a snake in my boot'
    
    print('input string: ', input_string)
    print('string as numbers', encode(input_string))
    print('output string: ', decode(encode(input_string)))
    
    # train and test data
    text_data = np.array(encode(text))
    cutoff = int(num_chars*0.9)
    text_data_train = text_data[0:cutoff]
    text_data_test = text_data[cutoff::]

    print('train shape: ', text_data_train.shape) 
    print('test shape: ', text_data_test.shape)        

    # Want to train on chunks of the text at a time. -- Blocks -- Block_size 
    block_size = 8 # maximum context length
 
    
    # so need minibacthes of text next so we can train on multiple sequences in parallel
    np.random.seed(8272025)

    def GetBatch(input_data, block_size=8, batch_size=32):
        ''' we want to have 2 arrays of size (batch_size, block_size)
        and we want them to be offset of eachother by 1, so that the input 
        is a sequence, and the output is the sequence 1 time step into the future.
        ''' 
        X_ = []
        Y_ = []
        # create a batch
        for i in range(batch_size):
            ix = int(len(input_data)*np.random.rand()) - block_size #dont reference anything at the end of the file
            # print('IX: ', ix)
            X_.append(input_data[ix:ix+block_size])
            Y_.append(input_data[1+ix:1+ix+block_size])
        return np.array(X_), np.array(Y_)
    
    # test it out on the training data. 
    X, Y = GetBatch(text_data_train, block_size=8, batch_size=4)   
    def convert_to_tensor(X,Y):
        return torch.from_numpy(X),torch.from_numpy(Y) 
    X, Y = convert_to_tensor(X,Y)
    print(X.shape)
    print(Y.shape)
    print(X)
    print(Y)

    ######
   
        
    Bmod =  BigramLanguageModel(vocab_size=vocab_size)
    out = Bmod (X, Y)
    print(out.shape)
