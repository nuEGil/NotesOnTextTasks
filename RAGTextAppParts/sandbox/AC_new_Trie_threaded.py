import re
import os
import glob
import json
import time
import threading
import queue
import numpy as np 
import pandas as pd 

'''
Extended trie to handle any character. memory is now variable.
base memory on node is 1 object rather than 26 Nones

Lets try multi threading this thing.

This doesnt actually run any faster than the single thread version. 
'''
def GetText(x):
    with open(x, "r", encoding='utf-8' ) as f:
        return f.read()

def CleanSpecialChars(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

class TrieNode():
    def __init__(self):
        self.stoptoken = False # conditional stop 
        # dict instead of list so we can index 1001 without 1001 entries
        self.subnodes = {}
    
class Trie():
    def __init__(self):
        self.root = TrieNode() # root object is a node itself... 
        self.wordlist = []
    
    def Insert(self, word):
        # Class Trie has a scope of memory, you will either make new TrieNode objects 
        # or you will reference an existing TrieNode - but the dict enables you to 
        # jump nodes
        
        curr = self.root # point to the root object
        for c in word: # itterate through the characters in the word            
            if not c in curr.subnodes.keys(): # check for the current letter in the node
                curr.subnodes[c] = TrieNode() # so then you make another TrieNode object and point to it with curr
        
            # Always point to the new subnode
            curr = curr.subnodes[c]

        # once you are all the way through the last node object can get the stop token.
        curr.stoptoken=True   
        self.wordlist.append(word) # keep track of all the words.           

    def Search(self, word):
        curr = self.root # initial TrieNode object
        for c in word:
            if not c in curr.subnodes.keys():
                return False # escape the loop and say the word isnt in there
            # at this point c is in the sub nodes
            curr = curr.subnodes[c]
        # we can only ever get to the end of the loop if all c are found in the set of
        # trie node objects
        return curr.stoptoken # So this is always going to return true if we get to the end. 
    
    def Prefix(self, word, delim = '*'):
        curr = self.root # initial TrieNode object
        for c in word:
            if not c in curr.subnodes.keys():
                return False # escape the loop and say the word isnt in there
            # at this point c is in the sub nodes
            if c == delim:
                return True           
            curr = curr.subnodes[c]
        # we can only ever get to the end of the loop if all c are found in the set of
        # trie node objects
        return curr.stoptoken # So this is always going to return true if we get to the end. 
    
    def PrefixTextSearch(self, text):
        # this will capture subwords as often as words... so likely dont do that
        #start a dictionary of lists to cary the word inds
        wordtracker = dict(zip(self.wordlist, [[] for j in range(len(self.wordlist))]))
        # loop through the text        
        curr = self.root # start at the root node
    
        buff = '' # start a text buffer
        for ti, tt in enumerate(text):
            buff+=tt # add the first character to the buffer 
            
            # if the character is not in the keys then go back to the root and reset the buffer
            if not tt in curr.subnodes.keys(): 
                curr = self.root  
                buff = '' 
            
            else:
                # The only other option is that the word is in here 
                curr = curr.subnodes[tt]

            # Exit condition
            if curr.stoptoken:
                # get the count and index
                wordtracker[buff].append(ti - len(buff))
                # reset
                curr = self.root # go back to the root if this character is not in there
                buff = ''
            
        return wordtracker

    def TextSearch(self, text, delims= [' ', '\n', '.', '!', '?', "'",'"']):
        # This version will break on delimiters
        wordtracker = dict(zip(self.wordlist, [[] for j in range(len(self.wordlist))]))
        buff = ''
        for ti, tt in enumerate(text):
            if not any(tt == dd for dd in delims):
                buff+=tt
            else:
                if self.Search(buff):      
                    wordtracker[buff].append(ti - len(buff))
                buff = ''   
        return wordtracker

def GetSubText(word_set, text, window = 400, ):
    # Testing out just a specific snippet of text
    # print('Searching text for any examples of key word')
    window = window // 2
    # itterate through the key words. 
    subdata = {}
    for k, v in word_set.items():    
        ids = []
        samples = []
        for ii, id in enumerate(v):  
            start_ = max(0, id - window) # avoid negative numbers 
            end_ = min(len(text), id + window) # avoid going out of bound on the text
            samples.append(text[start_ : end_])
            ids.append(id)
        subdata[k] = {'ids' : ids, 'samples' : samples}
    return subdata

def ThreadJob(q, mdata='', Trie_ = ''):
    ## likely to cause a bug in the multithread world. it's going to use a 
    # single word list and a single trie object for every thread
    data_part = []   
    for fi in range(mdata.shape[0]):
    # read the text file 
        text = GetText(mdata['title'].iloc[fi])
        sub_title = mdata['title'].iloc[fi].split('/')[-1].replace('.txt','')        
        # search for the words in word set 
        word_set = Trie_.TextSearch(text.lower())
        sub_dat = GetSubText(word_set, text, window = 200, )
        data_part.append({
                            'id':fi,
                            'name' : sub_title,
                            'word_set':sub_dat
                         })
    # add to the result queue
    q.put(data_part)
    print('job complete')

def GetBookMetadata(bookdir):
    names = sorted(glob.glob(os.path.join(bookdir, '*/*.txt')))
    metadata = {'title':names, 'source':[n.split('/')[-2] for n in names]}
    metadata = pd.DataFrame.from_dict(metadata)
    metadata.to_csv(os.path.join(bookdir, 'metadata.csv'))
    return metadata

if __name__ == '__main__':
    # start the timer
    start_time = time.time()

    bookdir = '/mnt/c/Users/gil82/Documents/books/'
    # check for metadata file 
    if not os.path.exists(bookdir+'metadata.csv'):
        print('No metadata file found... constructing')
        metadata = GetBookMetadata(bookdir)
    else:
        print('Loading found metadata.csv') 
        metadata = pd.read_csv(bookdir+'metadata.csv')
    # only work with the books 
    # metadata = metadata[metadata['source']=='books']
    print('metadata shape ', metadata.shape)

    wordlist = sorted(['love', 'passion', 'faith', 'war',
                       'time','seconds', 'minutes', 'days', 'weeks',
                       'months', 'years', 'hours', 
                       'mind', 'mental', 'memory', 'memories',
                       'think', 'thinking', 'thought', 'thoughts',
                       'prayers', 'pray', 'prayed', 'meditate',
                       'soul', 'spirit', 'dream', 
                       'body', 'bodies', 'blood', 'guts', 'spit', 
                       'vomit', 'excrement', 'soiled',
                       'eyes', 'mouth', 'nose', 'ears', 'hands',
                       'hand', 'fingers', 'finger', 'feet', 'toes',
                       'heart', 'stomach', 'nerve', 'nervous',
                       'neck', 'chest', 'breast', 'shoulders',
                       'kill', 'killed','murder',
                       'death', 'died', 'dead',
                       'drunk', 'drink', 
                       'drinking', 'smoking', 
                       'gambling', 'gamble','gambler', 'alcoholic',
                       'habit', 'drug', 'drugs','opium', 'cocaine',
                       'family','father', 'mother', 'brother', 'sister',
                       'aunt', 'uncle', 'cousin', 'son', 'daughter', 
                       ])
    print('number of words to track : ', len(wordlist))
    # set up the trie
    Trie_ = Trie()
    for w in wordlist:
        Trie_.Insert(w)
    
    # Set up how much data to use per thread
    n_threads = 3
    n_batch = metadata.shape[0] // n_threads
    n_res = metadata.shape[0] % n_threads 
    print(f'n_batch:{n_batch} n_residual:{n_res}')

    # now im pretty sure I can set up.. n separate trie objects. 
    results_queue = queue.Queue() # set up a queue
    threads = []
    for j in range(0, metadata.shape[0], n_batch):
        # make sure we get all the files
        if j + n_batch < metadata.shape[0]:
            sub = metadata.iloc[j:j+n_batch]
        else:
            sub = metadata.iloc[j::]

        # Using `args` to pass positional arguments and `kwargs` for keyword arguments
        t = threading.Thread(target=ThreadJob, 
                             args = (results_queue,),
                             kwargs={"mdata": sub,
                                     "Trie_":Trie_})
        threads.append(t)
    # Start each thread
    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    full_data_set = {}
    full_data = []  
    for lq in list(results_queue.queue):
        full_data.extend(lq)
    print(len(full_data))

    for id, val in enumerate(full_data):
        full_data_set[id] = val
        
    outfile = os.path.join(bookdir,'dataset/dataset.json')
    with open(outfile, 'w') as f:
        json.dump(full_data_set, f, indent = 4)

    end_time = time.time() # new time instance
    elapsed_time = end_time - start_time
    print(f"Threading time : {elapsed_time} seconds")