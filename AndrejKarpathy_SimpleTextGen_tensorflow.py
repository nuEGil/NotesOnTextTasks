import os 
import numpy as np 
import tensorflow as tf
from transformer_parts import *
'''follow this 
https://www.tensorflow.org/text/tutorials/transformer#data_handling

Skeleton implementation of this thing. 
'''
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

def GenerateTextViaTransformer(decoder, model, block_size,  max_new_tokens = 100):
    # in karpathy's example he was generating tokens in a lst of max new token
    print('generating text')
    # x_start = tf.convert_to_tensor(np.zeros((1,block_size)))
    # y_start = tf.convert_to_tensor(np.zeros((1,block_size)))
    x_start = np.zeros((1,block_size))
    y_start = np.zeros((1,block_size))
    token_sequence = [0]
    for ji in range(max_new_tokens):
        # the start token is 0 -> new line right
        output = model.predict([x_start, y_start])[0,-1,:] # only get the last logit right
        # print(type(output))        
        # # apply softmax on this thing so you now have a probability distribution 
        softmax_output = tf.nn.softmax(tf.convert_to_tensor(output))
        softmax_output = softmax_output.numpy().astype(np.float64)
        # print('softmax output shape ',softmax_output.shape)
        # print(softmax_output)
        # now sample the multinomial distribution --> soft max made a proability distribution, 
        # you want to get 1 semple from that distribution -- the length of the soft max will 
        # tell it the numbers -  so it probability of 0,vocabsize pretty much.
        new_token = np.argmax(np.random.multinomial(n = 1, pvals = softmax_output))
        # print('new token ', new_token)

        # append the new token
        token_sequence.append(new_token)

        # update the buffer for x and y start
        x_start=0+y_start # update to whatever y_start currently is
        # now put the new token into y _start
        y_start[0, 0:-1] =0+y_start[0, 1::]
        y_start[0, -1] = 0+new_token

        # print(x_start)
        # print(y_start)
    stringout = decoder(token_sequence) 
    print('string out : ', stringout)

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
 
    # predict at every point of the chunk -- the plus one is the last 
    # rememebr that it is an autoregressive model where the output is dependet on the past sequence of outputs
    # data loader is going to need something like this. --
    #  want transformer to be used to seeing everything from 1 to block_size length sequence
    #  after that you can run it in a loop right. 
     
    # X = text_data_train[0:block_size]
    # Y = text_data_train[1:block_size+1]
    # for t in range(block_size):
    #     context = X[:t+1] # previous chunk of outputs
    #     target = Y[t] # next token in the sequence. 
    #     print(f'when input is {context} the target: {target}')    
    
    # so need minibacthes of text next so we can train on multiple sequences in parallel
    np.random.seed(8272025)
    
    # test it out on the training data. 
    X, Y = GetBatch(text_data_train, block_size=8, batch_size=4)   
    print(X.shape)
    print(Y.shape)
    print(X)
    print(Y)

    ######
    # split off point in the tutorial. 
    # I want to make use of the TensorFlow for no
    num_layers = 2
    d_model = 64
    num_heads = 4
    dff = 256

    transformer = Transformer(
                        num_layers=num_layers,
                        d_model=d_model,
                        num_heads=4,
                        dff=dff,
                        input_vocab_size=vocab_size ,
                        target_vocab_size=vocab_size ,
                        dropout_rate=0.1)
    
    print(transformer.summary()) 
    # final layer output is this (batch_size, target_len, target_vocab_size)
    
    # ok so this part works... I just gotta figure out how to train now. 
    # this next one is the call
    transformer_input= [X,Y]
    output = transformer.predict(transformer_input)
    # so we are at the level of logits. The channels dimension is equal to the vocabulary size
    # so (Batch, Tokens, Channels) -> (B,T,C)
    print(output.shape)

    # now make the learning rate change on a schedule    
    # learning_rate = CustomSchedule(d_model)
    # make an adam optimizer object with that learning rate
    # transformer so the learning rate really matters -- its a lot of params
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, beta_1=0.9, 
                                        beta_2=0.98, epsilon=1e-9)
    
    # loss function is janky. In other examples we would just one hot encode Y and call it a day. 
    transformer.compile(loss = make_token_ce(vocab_size,), optimizer=optimizer, metrics = [masked_accuracy])

    GenerateTextViaTransformer(decode, transformer, block_size=block_size, max_new_tokens = 10)
    # X__ ,Y__ = GetBatch(text_data_train, block_size=8, batch_size= 512)
    # transformer.fit(x=[X__,Y__], y=Y__, epochs=10, batch_size = 32, )
    
    # Train the model 

    # work around -- get one really large batch and pass it to fit. it will figure it out;
    # the other thing is that this model is going to be large -- so you'll need a lot of data for 
    # these token sequences to be of any use 
    X__ ,Y__ = GetBatch(text_data_train, block_size=8, batch_size= 50000) 

    # I want to reshape Y to be (batch_size* block_size), and out  to be (batch_size* blocksize, logits)
    transformer.fit(x=[X__,Y__], y = Y__, epochs=10, batch_size = 128, )
    GenerateTextViaTransformer(decode, transformer, block_size=block_size, max_new_tokens = 100)