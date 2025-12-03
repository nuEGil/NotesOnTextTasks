import math
import torch
import numpy as np 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

'''Add in the token prior computation part. '''


class PositionalEncoding(nn.Module):
    def __init__(self, seq_length=512, dim=512, device = 'cpu'):
        super().__init__()
        dim_half = dim // 2
        positions = torch.arange(seq_length, device = device).unsqueeze(1)
        depths = torch.arange(dim_half, device = device).unsqueeze(0) / dim_half
        angle_rates = 1.0 / (10000 ** depths)
        angle_rads = positions * angle_rates
        self.encode_arry = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1)

    def forward(self, x):
        # pe = 
        x = x + self.encode_arry[:x.size(1), :].unsqueeze(0)
        return x
    
class MLP(nn.Module):
    def __init__(self,  neurons_in = 64, dropout_rate = 0.5, n_out = 64):
        super(MLP, self).__init__() # make sure the nn.module init funciton works

        # base neuron set - can edit this 
        self.neurs = [ (neurons_in, neurons_in*2),
                       (neurons_in*2, neurons_in)]
        
        # make a module list for these things -- dont have to spplit it up like this
        self.layers = nn.ModuleList() 
        
        # loop if we decide to add more intermediate layers
        for neur in self.neurs:
            # dense layers are also refered to as fully connected layers
            self.layers.append(nn.Linear(neur[0], neur[1]))

        # shape of object yo be normalized
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = nn.ReLU()
        # final layer
        self.final_lin = nn.Linear(neurons_in, n_out)
        
    def forward(self, x):
        # applies the layers to x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.final_lin(x)
        return x

class BaseAttention(nn.Module):
    def __init__(self, dim = 512, heads = 4):
        super(BaseAttention, self).__init__() # make sure the
        self.mha = nn.MultiheadAttention(embed_dim = dim, num_heads = heads, batch_first = True)
        self.layernorm = nn.LayerNorm(normalized_shape = dim)

    def forward(self, x):
        x1, attn_weights = self.mha(x, x , x)
        x1 = self.layernorm(x + x1)
        return x1, attn_weights 
    
class Encoder(nn.Module):
    def __init__(self, dim = 64, heads = 4):
        super().__init__()
        self.attn = BaseAttention(dim = dim, heads = heads)
        self.mlp1 = MLP(neurons_in = dim, dropout_rate = 0.5, n_out = dim)
        self.layernorm = nn.LayerNorm(normalized_shape = dim)

    def forward(self, x):
        x1, attn_weights = self.attn(x) # multihead self attention
        x2 = self.mlp1(x1) # then an mlp block
        out = self.layernorm(x2 + x1) # residual with a layer norm
        return out

class MyModel(nn.Module):
    def __init__(self, vocab_size = 32, seq_length = 32, 
                 dim = 64, heads = 4, n_out=32, device = 'cpu',
                 max_new_tokens = 256):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim) # Vocab is needed to make a dicitonary to map integers in the vocab to the vector space
        self.pos_encode = PositionalEncoding(seq_length = max_new_tokens, dim = dim, device=device)
        
        # self.enclist = [Encoder(dim = dim, heads = heads) for i in range(3)]
        self.encoder0 = Encoder(dim = dim, heads = heads)
        self.encoder1 = Encoder(dim = dim, heads = heads)
        # model output
        self.OMLP = MLP(neurons_in = dim, dropout_rate = 0.5, n_out = n_out)

    def forward(self, x, targets = None):
        # print('x shape sanity ', x.shape)
        x1 = self.embed(x)# first layer is positional embedding
        x1 = self.pos_encode(x1) # positional encoding. 
        # for i in range(3):
        #     x1 = self.enclist[i](x1)
        x1 = self.encoder0(x1)
        x1 = self.encoder1(x1)
        logits = self.OMLP(x1)
        
        if targets is None:
            loss = None

        else:
            # expects the channels to be the 2nd dimenssion
            B,T,C = logits.shape
            # print(f'BTC {logits.shape}')
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

if __name__ == '__main__':
    seq_len = 16
    vocab_size = 3200
    batch_size = 32
    dim_ = 64
    data = torch.tensor(vocab_size*np.random.rand(batch_size, seq_len), dtype = torch.int64)

    # pp = PositionalEmbedding(vocab_size = vocab_size, seq_length=seq_len, dim= dim_)
    pp = MyModel(vocab_size = vocab_size, seq_length = seq_len, 
                 dim = dim_, heads = 4, n_out = vocab_size)
    print(pp)
    logits, loss = pp(data)
    print('output shape ', logits.shape)
    print(logits.min(), logits.max())