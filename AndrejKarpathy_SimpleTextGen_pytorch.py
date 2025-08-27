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

def convert_to_tensor(X,Y):
    return torch.from_numpy(X).to(torch.long), torch.from_numpy(Y).to(torch.long) 
# Basic model for testing purposes

def GetBatch(input_data, block_size=8, batch_size=32):
    ''' we want to have 2 arrays of size (batch_size, block_size)
    and we want them to be offset of eachother by 1, so that the input 
    is a sequence, and the output is the sequence 1 time step into the future.
    
    Something is up with the way we are selecting the text sequences but whatever. 
    ''' 
    X_ = np.zeros((batch_size, block_size))
    Y_ = np.zeros_like(X_)
    # create a batch
    for i in range(batch_size):
        ix = np.random.randint(0, len(input_data)-block_size)
        # print('IX: ', ix)
        X_[i,...]= 0+input_data[ix:ix+block_size]
        Y_[i,...]= 0+input_data[1+ix:1+ix+block_size]
    return np.array(X_), np.array(Y_)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token froma lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            # expects the channels to be the 2nd dimenssion
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T) 
            loss =  F.cross_entropy(logits, targets) 
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get preditctions 
            logits, loss = self(idx)
            # focus on only the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim = -1) # (B,C)
            # sample from the distributions 
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            # appends sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim =1) #(B, T+1)
        return idx

def ModelCheck(text_data_train):
    # test it out on the training data. 
    X, Y = GetBatch(text_data_train, block_size=8, batch_size=32)   
    X, Y = convert_to_tensor(X,Y)
    print(X.shape)
    print(Y.shape)
    print(X)
    print(Y)
    # so the same thing. in the tutorial it comes out as (4,8,65) so vocab size - cause he uses tiny shakespeare
    # this only runs if you don't define the loss. 
    Bmod =  BigramLanguageModel(vocab_size=vocab_size)
    logits, loss = Bmod (X, Y)
    print(logits.shape)
    print(loss) # want roughly -ln(vocab size - 89) - in video he had 5 and its close to 4.7 so all good.


    # Want to be able to generate from the model
    # batch and time will be 1, its a 1x1 tensor, datatype integer -- 0 is the new line character.     
    # ask for 100 new tokens
    tokens_oout = Bmod.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)
    tokens_oout = decode(tokens_oout[0].tolist())
    print(tokens_oout) # prints out nonsense before training. 
    return Bmod

def modeltraining(model, text_data_train, block_size=8, batchsize = 32, N_steps = 1000):
    ##### Train the model  
    # create a pytorch optimizer -- Adam is adaptive, high learning rate can be used for smaller models.
    # smaller learning rates like 1e-6 must be used for larger models.  hyper param tuning though
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for steps in range(N_steps):
        # sample a batch of data 
        X, Y = GetBatch(text_data_train, block_size=block_size, batch_size=batchsize)   
        X, Y = convert_to_tensor(X,Y)

        # evaluate the loss 
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if steps%100 == 0:
            print(f'steps: {steps}, training loss: ', loss.item())
    print('steps: training loss: ', loss.item())

    # predict again but not shakespear
    tokens_oout = Bmod.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)
    tokens_oout = decode(tokens_oout[0].tolist())
    print(tokens_oout) # prints out nonsense before training. 

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
 
    
    # check to see if you can get anything out of the model 
    Bmod = ModelCheck(text_data_train)

    # train the model .. done. -- at the end of training you'll get some less nonsense text 
    # easy to add a step to get validation loss and plot the 2 but we have all the major pieces. 
    modeltraining(Bmod, text_data_train, 
                  block_size=8, batchsize = 32,
                  N_steps = 50000)
    

    
    
    