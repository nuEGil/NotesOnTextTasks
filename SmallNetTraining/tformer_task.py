import os
import pandas as pd
from tokenizers import *
from tformer_arch import * 
from torch.utils.data import Dataset, DataLoader

class CustomTextDataset(Dataset):
    '''Only need to define logic for getting 1 item - loader itterates through the ful set'''
    def __init__(self, odata, text, sequence_length = 32):
        self.sequence_length = sequence_length
        print('Text length :', len(text))
        # make the token set
        tokens = np.array(odata['encoder'](text))
        # pad the token set so we can get sequences
        pad_char = odata['string_to_int']['  ']
        n = len(tokens)
        pad_len = (-n) % sequence_length  # how many extra tokens to add
        print('pad char ', pad_char)
        if pad_len > 0:
            tokens = np.pad(tokens, (0, pad_len), mode='constant', constant_values=pad_char)

        # get sequences 
        self.token_set = np.array([tokens[ii:ii+sequence_length] for ii in range(0, len(tokens), sequence_length)])

        print('token set shape : ', self.token_set.shape)

    def __len__(self):
        return len(self.token_set)-1

    def __getitem__(self, idx):
        # getting a single item should be this easy I think 
        X_ = self.token_set[idx]
        Y_ = self.token_set[idx+1]
        
        return X_, Y_
        
def modeltraining(odir, odata, 
                  device, model, loader, optimizer,
                  N_steps = 1000,
                  tag = 0,
                  log_ = {'step':[], 'training_loss':[]},
                  max_new_tokens = 256):
    
    for steps in range(N_steps):
        # sample a batch of data 
        total_loss = 0
        for X,Y in loader:
            ## sanity check on what ever x data is feeding the network. 
            # X is (batch, sequence length) right . so I want to decode it 
            # for gg in range(3):
            #     print('xsample :', ''.join(odata['decoder'](X[gg].numpy())))

            # send data to GPU
            X = X.to(device)
            Y = Y.to(device)

            # evaluate the loss 
            logits, loss = model(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f' intermediate training loss: ', loss.item())

            total_loss += loss.item()

        print(f"Epoch [{steps+1}/{N_steps}], Loss: {total_loss/len(loader):.4f}")
        # print(f'steps: {steps}/{N_steps}, training loss: ', loss.item())
        log_['step'].append(steps)
        log_['training_loss'].append(total_loss/len(loader))
        log0 = pd.DataFrame.from_dict(log_)
        log0.to_csv(f'{odir}/log-{tag}.csv')
    
        if steps%100 == 0:
            # predict tokens out -- this time though make sure the tensor is on the gpu
            tokens_oout = model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=max_new_tokens)
            tokens_oout = odata['decoder'](tokens_oout[0].tolist())
            print(f'--- prediction post training for {steps} ---')
            print(tokens_oout) # prints out nonsense before training. 
            torch.save({
                        'epoch': steps,
                        'model_state': model.state_dict(),
                        # 'optimizer_state': optimizer.state_dict(),
                    }, f'{odir}/mod_tag-{tag}_steps-{steps}.pt')

             
    # save final step
    # predict tokens out -- this time though make sure the tensor is on the gpu
    tokens_oout = model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=max_new_tokens)
    tokens_oout = odata['decoder'](tokens_oout[0].tolist())
    print(f'--- prediction post training for {steps} ---')
    print(tokens_oout) # prints out nonsense before training. 
    torch.save({
                'epoch': steps,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, f'{odir}/mod_tag-{tag}_steps-{steps}.pt')

    return log_, model, optimizer

if __name__ =='__main__':
    odir = '/mnt/f/data/models/bigram/bugcheck_encoder'
    # read text
    text_dir = "/mnt/f/data/ebooks_public_domain/"
   
    #--- Model initialization ---#
    # model training arguments
    sequence_length = 32 # max sequence length 
    embed_dim = 256 # number of features to use in the embeding
    batch_size = 512
    max_new_tokens = 128

    # --- special load common words first ---#
    # load vocabulary set 
    print('--- load vocabulary set---')
    odata = LoadVocabularySet(gram_='bigram')

    # data loader initialization 
    # text = GetText(text_dir + "wordlist.10000.txt")
    # text = text.replace('\n', ' ')

    text = GetText(text_dir + "Salome_(Wilde_1904).txt")
    # special first data set 
    data_set = CustomTextDataset(odata, text, sequence_length=sequence_length)
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
 
    # --- Now model train ---#
    # Train model on word list.
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"------ Using device: {device} ------")

    # send the model to the device    
    model =  MyModel(vocab_size=odata['vocabulary_size'],  
                     seq_length = sequence_length,
                     dim=embed_dim,
                     heads = 4,
                     device=device,
                     n_out = odata['vocabulary_size'],
                     max_new_tokens = max_new_tokens).to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    modeltraining(odir, odata, 
                    device, model, loader, optimizer,
                    N_steps = 1000,
                    tag='salome',
                    max_new_tokens = max_new_tokens)

    