import re
import os
import glob
import json
import time
import asyncio
import sqlite3 
import datetime 

class DataEntry():
    # data structure to hold information on each text message
    def __init__(self, LineId:int, Text:str, 
                 Wordset:list, TimeStamp:str):
        self.LineId = LineId 
        self.Text = Text
        self.Wordset = Wordset 
        self.TimeStamp = TimeStamp

    def ToString(self):
        print(f"LineId:{self.LineId}\nTxt:{self.Text}\nWordset:{self.Wordset}")

    def ToDict(self):
        return {'LineId':self.LineId, 'Text':self.Text, 
                'Wordset':self.Wordset, 'TimeStamp':self.TimeStamp}

# both have to be asynchronous I think
async def LineParser(filename:str, wordlist:set):
    nminutes = 0
    ff = open(filename, "r", encoding = "utf-8")
    DataEntryList = []
    line_count = 0
    for line in ff:
        txt = ff.readline() 
        line_count+=1
        wordset = [wl for wl in wordlist if wl in txt] 
        
        if len(wordset)>=1:
            # simulating time passage
            timestamp_ = datetime.datetime.now() + datetime.timedelta(minutes = nminutes)
            timestamp_ = timestamp_.strftime("%Y-%m-%d %H:%M:%S")
            new_entry = DataEntry(line_count, txt, wordset, timestamp_)
            DataEntryList.append(new_entry)

            nminutes+=1
    ff.close()
    # print(f"File: {filename}  linecount:{line_count}")
    return DataEntryList, filename

def WriteJSON(results, output_dir):
    # now we can save to a JSON file if we want to do that. 
    full_data = dict()
    for ir,r in enumerate(results):
        full_data[ir] = {'filename':r[1],'data':{j:r_.ToDict() for j,r_ in enumerate(r[0])}}

    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        # formatting with indent takes up more space but its human readable
        json.dump(full_data, f, indent = 4)  

def WriteSQL(results, output_dir):
    # sql database  create tables
    con = sqlite3.connect(os.path.join(output_dir,"textdata.db"))
    cur = con.cursor() # createa cursor object
    cur.execute("""
                CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE);
                """)

    cur.execute("""
                CREATE TABLE entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                line_id INTEGER,
                text TEXT,
                wordset TEXT, -- store JSON string here
                timestamp TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id));
                """)

    # Add in entries
    for ir, r in enumerate(results):
        cur.execute("INSERT OR IGNORE INTO files (filename) VALUES (?)", (r[1],))

        for j,entry in enumerate(r[0]):
            # get file id
            cur.execute("SELECT id FROM files WHERE filename=?", (r[1],))
            file_id = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO entries (file_id, line_id, text, wordset, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (file_id, entry.LineId, entry.Text, json.dumps(entry.Wordset), entry.TimeStamp))

    con.commit() # only commit at the end. to avoid slow performance
    con.close()  # close the sonnection to the database. 

async def InitializeDatabase():
    start_time = time.time() # start time 
    # base set of tags. 
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
    base_dir = os.environ['BOOK_DIR']
    book_dir = os.path.join(base_dir, "books/")
    output_dir = os.path.join(base_dir, "outputs/") 
    print('env dir = ', book_dir)

    filenames = glob.glob(os.path.join(book_dir, '*.txt')) # this is a different list than the next one
    
    # open the file and read each line one by one. 
    tasks = [asyncio.create_task(LineParser(fname, wordlist)) for fname in filenames ] 
    results = await asyncio.gather(*tasks) # passes each element of the list not the list
    
    # initial end time to match run time of the other scripts. 
    book_time = time.time()    
    print(f"Book processing time : {book_time - start_time}")
    WriteJSON(results, output_dir)
    json_time = time.time()
    print(f"json processing time : {json_time - book_time}")

    if not os.path.exists(os.path.join(output_dir, "textdata.db")):
        WriteSQL(results, output_dir)
        sql_time = time.time()    
        print(f"SQL time : {sql_time - json_time}")
    
    end_time = time.time()
    print(f"total time : {end_time - start_time}")

if __name__ == '__main__':
    asyncio.run(InitializeDatabase())