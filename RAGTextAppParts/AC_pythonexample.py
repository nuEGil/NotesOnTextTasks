'''Implementing a Trie just to make sure we have the gist of it 
ok but now you can run it on the full text or run it on a sentence window. interesting
ok so this implementation takes like 450ms, while the c++ code take ~200ms
so like... not a totally big deal. 

got it to 160ms by making a function....maybe cause its pycached? so 2nd time running? 
idk

this can work for phrases too. so might be a good idea to experiment with that. 
'''

import os
import json
import time

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

def ProcessTextWTrie(myTrie, text, wordcounts_, savetag = True):
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
                wordcounts_[buff].append(ti-len(buff))

            # reset the buffer
            buff = ''    
    # save file 
    if savetag:
        with open("py_word_trie.json", "w") as f:
            json.dump(wordcounts_, f, indent=4)
    return wordcounts_

if __name__ == '__main__':
    # start time 
    start_time = time.time()
    
    # read text
    text = GetText("crime and punishment.txt").lower()
    
    # words 
    wordlist=[
                "rodion", "pulcheria", "alexandrovna", 
                "dounia", "raskolnikov", "romanovitch", 
                "porfiry", "pyotr", "petrovitch",
                "dmitri", "razumihin","prokofitch", "sofya", "semyonovna", 
                "marmeladov","amalia", "fyodorovna",
                "lebeziatnikov","darya","frantsovna", 
                "katerina", "ivanovna", "fyodor", "dostoyevsky",
                "dostoevsky", "svidrigailov"
                ]
    wordcounts = dict(zip(sorted(wordlist), [[] for w in wordlist]))
    

    # build the Trie
    myTrie = Trie()
    # but now you cant print it. 
    [myTrie.insert(word) for word in wordlist]
    
    # now process the text 
    wordcounts = ProcessTextWTrie(myTrie, text, wordcounts)

    print(wordcounts)
    print(text[wordcounts['alexandrovna'][0]:wordcounts['alexandrovna'][0]+len('alexandrovna')])
     
    end_time = time.time()
    print("Elapsed : ", end_time - start_time, "seconds")
    
