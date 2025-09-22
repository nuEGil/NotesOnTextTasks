'''Implementing a Trie just to make sure we have the gist of it 
ok but now you can run it on the full text or run it on a sentence window. interesting
ok so this implementation takes like 450ms, while the c++ code take ~200ms
so like... not a totally big deal. 

got it to 160ms by making a function....maybe cause its pycached? so 2nd time running? 
idk

this can work for phrases too. so might be a good idea to experiment with that. 
'''
import re
import os
import json
import time
import numpy as np 

def GetText(x):
    with open(x, "r", encoding='utf-8' ) as f:
        return f.read()
    
class TrieNode():
    def __init__(self):
        # You make an empty list for every letter in the alphabet 
        self.subnodes = [None] * 26
        self.isEndOfWord = False

class Trie():
    def __init__(self):
        # start a root node
        self.root = TrieNode()
    
    def insert(self, key):
        # this will add keys in - so 1 element of word list 
        curr = self.root
        # check each character in the word
        for c in key:
            # the index really is the distance from the letter a 
            # so make sure that the thing is all lower case
            index = ord(c) - ord('a')
            
            # reference the subnode list and if it is none make a new node
            # on the first itteration it will always be none, so by default 
            # you make a node
            if curr.subnodes[index] is None:
                curr.subnodes[index] = TrieNode() # make a new node
            
            # A new node has been made right, so now its a TrieNode class.  
            curr = curr.subnodes[index]   

        # the last node you get to should have end of word    
        curr.isEndOfWord = True
    
    def search(self, key):
        # See if this whole word is stored in the key
        curr = self.root
        # itterate through the characters 
        for c in key:
            index = ord(c) - ord('a')
            if curr.subnodes[index] is None:
                return False # thing isnt in there.
            curr = curr.subnodes[index]
        return curr.isEndOfWord
    
    # Method to check if a prefix exists in the Trie
    def isPrefix(self, prefix):
        curr = self.root
        for c in prefix:
            index = ord(c) - ord('a')
            if curr.subnodes[index] is None:
                return False
            curr = curr.subnodes[index]
        return True

def ProcessTextWTrie(myTrie, text, wordcounts_):
    # if you print the subnodes, then youll just end up showing the class not the character. 
    # ok lets use this thing to scan the text.... 

    buff = ''
    for ti, tt in enumerate(text):
        
        # ascii value for letters doesnt start with 0. 
        index_ = ord(tt) - ord('a') # so get the difference to get the alphabet index
        bool_ = index_ <26 and index_>=0 # is the character in range
        # print(tt, ord(tt) - ord('a'), bool_)       
        if bool_:
            # if the character is in range cool build a word
            buff+=tt
        
        else:
            # as soon as the character is not in range then 
            # check it out
            # print(ti,'BUFFFF', buff, myTrie.search(buff))
            if myTrie.search(buff):
                wordcounts_[buff]['char_ids'].append(ti-len(buff))

            # reset the buffer
            buff = ''    
       
    return wordcounts_

def GetSentences0(text, offset = 0):
    # get any character followed by a group of chatacters till you get 
    # to a terminating puncuation
    text = text.replace('\n',' ')
    pattern = r".+?[.?!](?=\s)"
    matches = re.finditer(pattern, text)

    sentence_starts = []
    sentence_ends = []
    sentences = []
    for m in matches:
        start_ = m.start()+offset
        stop_ = m.end()+offset
        sentence_starts.append(start_)
        sentence_ends.append(stop_)
        sent = f"'{m.group()}' [{start_}:{stop_}]"
        sentences.append(m.group())
        # print(sent)

    return sentence_starts, sentence_ends, sentences

def CleanSpecialChars(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def GetSinglePlayerSubtexts(text, N_sentences, sentence_starts, 
                            sentence_ends, wordcounts, window_length = 8):
        
    # stride of window length to make this thing go a bit faster and save less data. 
    # 10 sentences cause why not. 
    for sentence_idx in range(0, N_sentences - window_length, window_length):
        # for sentence_idx in range(len(sen))
        # so the next thing we want. 
        # sentence_idx = 500 # ge the 20th sentence 
        window_start = sentence_starts[sentence_idx]
        window_end = sentence_ends[sentence_idx + window_length]
        sub_text = CleanSpecialChars(text[ window_start : window_end ])

        # Next append to the sentence ids
        for k,v in wordcounts.items():
            v = np.array(v['char_ids'])
            sub_v = v[(v>=window_start) & (v<=window_end)]
            if len(sub_v) > 0:
                # print(k, sub_v) 
                wordcounts[k]['sentence_ids'].append(sentence_idx)
                wordcounts[k]['sub_text'].append(sub_text)
    
    return wordcounts

def GetImportantPairs0(LikelyPlayers):
    # take all the players from the dict and make pairs     
    npDC_2 = np.array(LikelyPlayers) # source nodes

    # number of players
    n = len(npDC_2)

    # Pairing players makes a square matrix - O(N^2) scaling so filter likely players
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask0 = i<j
    pairs = np.column_stack((npDC_2[i[mask0]], npDC_2[j[mask0]])) # branches
    pairs_counts = np.zeros((pairs.shape[0],))
    print(pairs.shape, n)
    print(pairs[0:10])
    print(npDC_2)
    # return the resulting player list, and the pairs
    return npDC_2, pairs, pairs_counts

def GetPairSubtexts(text, N_sentences, sentence_starts, 
                    sentence_ends, wordcounts, pairs,  window_length = 8):
        
    # stride of window length to make this thing go a bit faster and save less data. 
    # 10 sentences cause why not. 
    pair_subtexting = {f'{a} : {b}':{'char_start':[] , 'char_end':[], 
                                    'sentence_id':[], 'sub_text':[],
                                    } for (a, b) in pairs}
    for sentence_idx in range(0, N_sentences - window_length, window_length):
        # for sentence_idx in range(len(sen))
        # so the next thing we want. 
        # sentence_idx = 500 # ge the 20th sentence 
        window_start = sentence_starts[sentence_idx]
        window_end = sentence_ends[sentence_idx + window_length]
        sub_text = CleanSpecialChars(text[window_start : window_end])

        for (a, b) in pairs:
            if (a in sub_text.lower()) and (b in sub_text.lower()):
                pair_subtexting[f'{a} : {b}']['char_start'].append(window_start)
                pair_subtexting[f'{a} : {b}']['char_end'].append(window_end)
                pair_subtexting[f'{a} : {b}']['sentence_id'].append(sentence_idx)
                pair_subtexting[f'{a} : {b}']['sub_text'].append(sub_text)
                
    return pair_subtexting


if __name__ == '__main__':
    # start time 
    start_time = time.time()
    
    # read text
    input_file = "/mnt/f/data/ebooks_public_domain/crime and punishment.txt"
    text = GetText(input_file)
    
    # words 
    wordlist=sorted([
                    "rodion", "pulcheria", "alexandrovna", 
                    "dounia", "raskolnikov", "romanovitch", 
                    "porfiry", "pyotr", "petrovitch",
                    "dmitri", "razumihin","prokofitch", "sofya", "semyonovna", 
                    "marmeladov","amalia", "fyodorovna",
                    "lebeziatnikov","darya","frantsovna", 
                    "katerina", "ivanovna", "fyodor", "dostoyevsky",
                    "dostoevsky", "svidrigailov"
                ])
    
    # word count dictionary.... 
    wordcounts = dict(zip(wordlist, [{'char_ids':[], 'sentence_ids':[], 'sub_text':[]} for w in wordlist]))
    
    # build the Trie
    myTrie = Trie()
    # but now you cant print it. 
    [myTrie.insert(word) for word in wordlist]
    

    # now process the text 
    wordcounts = ProcessTextWTrie(myTrie, text.lower(), wordcounts)
    print(wordcounts.keys())
    # lets say we have the words already. we want a fast way of checking if pairs of words occured together right.. 
    # get sentences 
    sentence_starts, sentence_ends, sentences = GetSentences0(text, )
    print(f'num characters : {len(text)} last sentence [{sentence_starts[-1]} : {sentence_ends[-1]}]')
    print(f'num sentences : {len(sentence_ends)}')
    
    
    window_length = 8
    wordcounts = GetSinglePlayerSubtexts(text, len(sentences), sentence_starts, sentence_ends,
                                        wordcounts, window_length)
    npDC_2, pairs, pairs_counts = GetImportantPairs0(wordlist)


    pair_subtexting = GetPairSubtexts(text, len(sentence_ends), sentence_starts, 
                                      sentence_ends, wordcounts, pairs,  window_length = 8)
    
    # filter the pairs
    pair_subtexting = {k:v for k,v in pair_subtexting.items() if len(v['char_start'])>10}
    

    end_time = time.time()
    print("Elapsed : ", end_time - start_time, "seconds")


    # write jsons so we can do something with it later. 
    with open("py_word_trie.json", "w") as f:
        json.dump(wordcounts, f, indent=4)

    with open("pair_stuff.json", "w") as f:
        json.dump(pair_subtexting, f, indent=4)

    