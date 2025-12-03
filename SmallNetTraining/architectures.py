import math
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

'''
Define architectures and architecture components here. 
'''

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, emb_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, emb_dim)
        """
        seq_len = x.size(1)
        # positions: 0, 1, 2, ..., seq_len-1
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embeddings = self.pos_emb(positions)  # (1, seq_len, emb_dim)
        return x + pos_embeddings

class ResidualConvBlock(nn.Module):
    def __init__(self, emb_dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(emb_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, E) = (batch, sequence, features)
        identity = x+0
        out = x.transpose(1, 2)   # (B, E, T)
        out = self.conv(out)
        out = out.transpose(1, 2) # (B, T, E)
        out = self.norm(out)
        out = self.act(out + identity)  # add residual

        return out

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
        self.LayerNorm = nn.LayerNorm(neurons_in)
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
        x = self.LayerNorm(x)
        x = self.final_lin(x)
        return x

# there's no softmax here. so remember to do argmax(softmax(output)) on classification 
class MyModel(nn.Module):
    def __init__(self, vocab_size, prior, seq_len = 16, emb_dim=64, n_out=10):
        super().__init__()
        
        # embeddings         
        self.embed = nn.Embedding(vocab_size, emb_dim)
        # self.pos_embed = LearnedPositionalEmbedding(seq_len, emb_dim)

        self.res = ResidualConvBlock(emb_dim=emb_dim, kernel_size = 7)
        # Initialize MultiheadAttention module
        self.multihead_attn = nn.MultiheadAttention(embed_dim = emb_dim, num_heads = 4)
        self.mlp = MLP(neurons_in=emb_dim, n_out=n_out)

        #--- define custom embedding weights with the prior ---#
        # use token probs 
        probs_t = torch.tensor(prior, dtype = torch.float32)
        probs_t = probs_t / probs_t.sum()
        # Base normal init (mean 0, std 0.02 like GPT-style)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        # Bias strength (you can tune this)
        bias_scale = 1
        # Compute a per-token bias
        # e.g., higher probability → slightly larger initial mean magnitude
        bias = (probs_t - probs_t.mean()) * bias_scale  

        # Broadcast bias across embedding dimensions
        self.embed.weight.data += bias.unsqueeze(1)
            
    def forward(self, x, targets=None):
        # x: (B, T)
        seq_out = self.embed(x)          # (B, seq, channels)
        # seq_out = self.pos_embed(seq_out)   # newly added positional embedding.
        seq_out = self.res(seq_out)     # (B, seq, channel)
        # self attention - no idea on the shape of the output yet
        attn_output, attn_output_weights  = self.multihead_attn(seq_out, seq_out, seq_out) 
        logits = self.mlp(attn_output)      # parsed by mlp
        # print('logit shape ', logits.shape)
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