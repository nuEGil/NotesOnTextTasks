import re
import os
import glob
import json
import time
import asyncio
import numpy as np 

class DataEntry():
    def __init__(self, LineId:int, Text:str, Wordset:list):
        self.LineId = LineId 
        self.Text = Text
        self.Wordset = Wordset 

    def ToString(self):
        print(f"LineId:{self.LineId}\nTxt:{self.Text}\nWordset:{self.Wordset}")

    def ToDict(self):
        return {'LineId':self.LineId, 'Text':self.Text, 'Wordset':self.Wordset}
    
# both have to be asynchronous I think
async def LineParser(filename:str, wordlist:set):
    # print('wordlis id ', id(wordlist))
    ff = open(filename, "r", encoding = "utf-8")
    DataEntryList = []
    line_count = 0
    for line in ff:
        txt = ff.readline() 
        line_count+=1
        wordset = [wl for wl in wordlist if wl in txt] 
        
        if len(wordset)>=1:
            new_entry = DataEntry(line_count, txt, wordset)
            DataEntryList.append(new_entry)
        
    ff.close()
    # print(f"File: {filename}  linecount:{line_count}")
    return DataEntryList, filename

async def main():
    # start time 
    start_time = time.time()
    wordlist = sorted(['love', 'passion', 'faith', 'war'
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
    wordlist = set(wordlist) # use a set for "in" comparisons - it should go faster
    print('number of words to track : ', len(wordlist))
    
    # Thats gonna be the next part. dont ever load all the book names into memory like that. 
    # always read one line at a time right. 
    basedir = '/mnt/c/Users/gil82/Documents/books/books/'
    filenames = glob.glob(os.path.join(basedir, '*.txt')) # this is a different list than the next one
    
    # open the file and read each line one by one. 
    tasks = [asyncio.create_task(LineParser(fname, wordlist)) for fname in filenames ] 
    results = await asyncio.gather(*tasks) # passes each element of the list not the list
   
    full_data = dict()
    for ir,r in enumerate(results):
        full_data[ir] = {'filename':r[1],'data':{j:r_.ToDict() for j,r_ in enumerate(r[0])}}

    with open('dataset.json', 'w') as f:
        json.dump(full_data, f, indent = 4)

    end_time = time.time()
    
    print(f"time elapsed:{end_time - start_time}")

if __name__ == '__main__':
    asyncio.run(main())