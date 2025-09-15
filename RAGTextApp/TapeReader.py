import os 
import re
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
'''Need to merge this with FeatureCrafting at some point. 
get words per sentence 
get sentence per paragraph
Look into some other features. -- because then we can use scikit learn to do 
like a bag of features or a decision tree and go nuts with old fashioned AI 
to detect key points in the text. 

Inspired by this youtube viedo https://www.youtube.com/watch?v=j2T4gvQAiaE&t=567s
-- Make a graph by counting the number of times that two nouns apear together
-- Take the double caps words and make a square matrix(dictionary of bigrams)
-- For every sentence (or window = 2, 3, 4, sentences) how many times did that pair come up (non adjacent) 
-- Normalize counts to get relationship score at every pair 
-- Monitor relationship score for interesting pairs - looking for changes before and after points in time
 
When you do move over to C or C++
-- Hash the text to get fixed length strings for every word / sentence / paragraph
-- -- key words like the book players, should be hashed when you're trying to match them
-- make trees where you can for matching things. 
-- use a text file with commonly used words to filter out things.
-- graphs scale O(n^2)

Might need to put in a list of key words and phrases... might not be a way around it. 
So like marmeladov is a pivitol character in crime and punishment, but his name only 
comes out like once. Hes a drunk and gets hit by a horse cart, dies, and thats 
what pushes his family further into poverty, and pushes raskolnikov and sonia(?)
together. Its one of the catalysts for Raskolnikov turning himself in. 


Make a JSON file for pair counts, and include the sentence index where the pairs happen.  so like 
name, name, list of inds.... 

Need to think of getting words that are within a window of sentences. because Raskolnikov and Dmitri never show together

'''


def GetText(x):
    with open(x, "r" ) as f:
        return f.read()

class TextObj():
    # this could be, sentence, paragraph, quoted speech
    def  __init__(self, text: str, start_idx: int, end_idx: int = None, label: str = None):
        self.text = text
        self.idx = start_idx
        self.end_idx = end_idx if end_idx is not None else start_idx + len(text)
        self.label = label  # optional: "word", "sentence", "ellipsis", etc.

def GetCharacters(text):
    # get Unique characters 
    chars = sorted(list(set(text)))
    letters = sorted([c for c in chars  if c.isalpha()]) 
    special_chars = "".join(sorted(list(set(chars) - set(letters))))
    return chars, special_chars

def GetParagraphs(text):
    # get all the paragraph indices.
    paragraph_inds = [0]
    for match in re.finditer(r'\n\n',text):
        paragraph_inds.append(match.start())
    
    paragraphs = []
    for pi in range(len(paragraph_inds)-1):
        paragraphs.append((text[paragraph_inds[pi]+2:paragraph_inds[pi+1]]))

    return paragraph_inds, paragraphs

def GetBoundText(text, open_, close_):
    # variant of get paragraphs
    # find all matches between double quotes
    # Escape in case delimiters are regex special chars
    open_esc = re.escape(open_)
    close_esc = re.escape(close_)
    
    pattern = f"{open_esc}(.*?){close_esc}"
    matches = list(re.finditer(pattern, text, flags=re.S))
    
    bound_inds = []
    bound_texts = []
    
    for m in matches:
        bound_inds.append(m.start())
        clean_quote = m.group(1).replace('\n', ' ')
        bound_texts.append(clean_quote)  # inside the delimiters
    
    return bound_inds, bound_texts

def GetSentences(text, offset = 0):
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

def GetSentencesFromParagraphs(paragraphs, paragraph_inds):
    # Get all the sentences in paragraphs
    sentence_starts = []
    sentence_ends = []
    sentences = []
    for para, pis in zip(paragraphs, paragraph_inds):
        a,b,c = GetSentences(para, offset = pis)
        sentence_starts.extend(a)
        sentence_ends.extend(b)
        sentences.extend(c)

    return sentence_starts, sentence_ends, sentences 

def GetDoubleCaps(text):
    text = text.replace('\n', ' ')
    # trying to find names here
    pattern = r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
    matches = list(re.finditer(pattern, text))
    
    start_inds =[]
    end_inds = []
    objs = []
    for m in matches:
        objs.append(m.group())
        start_inds.append(m.start())
        end_inds.append(m.end())
    return start_inds, end_inds, objs

def GetWordsAndInds(text, special_chars = '.'):
    pattern = "[" + re.escape(special_chars) + "]"
    # replace the special character with a space
    f_text = re.sub(pattern, " ", text)
    
    # now you have every word in order of occurence 
    words = f_text.split()
    total_n_words = len(words)
    uwords = set()
    uwords_and_locs = []
    for wi, word_ in enumerate(words):
        # now you have an index and each word
        # I want to know that it is alpha numeric
        filtered_word = ''.join([fw_ for fw_ in word_.lower() if fw_.isalpha()])
        if not filtered_word in uwords:
            uwords.add(filtered_word)        
            uwords_and_locs.append([filtered_word,wi])
    return uwords_and_locs, total_n_words

def GetWordCounts(X):
    X_counts = {}
    for x in X:
        if not x in X_counts.keys():
            X_counts[x] = 1
        else:
            X_counts[x]+=1
    return X_counts

def GetLikelyPlayers(DoubleCapped_):
    # Get the individual words 
    DC_2 = []    
    for dc in DoubleCapped_:
        DC_2.extend(dc.split())
    
    DC_2_counts = GetWordCounts(DC_2)  
    sorted_items_descending = sorted(DC_2_counts.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_items_descending)
    
    # filter this so you dont get an unmanageable amount of word pairs
    thresh = 15
    common = ['Project','Gutenberg', 'The', 'But', 'And']
    DC_2_counts['Marmeladov']+=20 # I know marmeladov is a character in here
    top_DC_2 = {k:v for k,v in DC_2_counts.items() if v>thresh and not any([k== c for c in common])}  
    
    # then this part is just for printing
    sorted_items_descending = sorted(top_DC_2.items(), key=lambda item: item[1], reverse=True)
    print(sorted_items_descending)
    
    # getting back a dictionary of the top players + their counts
    return top_DC_2

def GetImportantPairs(LikelyPlayers):
    # take all the players from the dict and make pairs     
    npDC_2 = np.array(list(LikelyPlayers.keys())) # source nodes

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

def CleanSpecialChars(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def word_to_int(words):
    # use this to convert words to integers in numpy 
    # quick text matcher through multiplication -- check for perfect squares
    return np.array([abs(hash(w.lower())) % 10_000_019 for w in words])

def perfect_square_matches(arr1, arr2):
    """
    Compare two arrays of words elementwise.
    Return a boolean mask where matches form perfect squares.
    """
    a = np.array([word_to_int(w) for w in arr1])
    b = np.array([word_to_int(w) for w in arr2])

    products = a * b
    sqrt = np.sqrt(products)
    return np.isclose(sqrt, np.round(sqrt))

def DoSomething(node_list, pairs, sentences, sentence_window = 10):
    running_char_set = []
    
    for sii in range(0,len(sentences)-sentence_window, sentence_window):
        sents_ = ' '.join(sentences[sii: sii + sentence_window])
        print(sents_)    
        # so the next thing thats going to happen is you're going to 
        # search through all the sentences... If say the first word
        # in the word pair isnt in there, then you can skip over the entire branch man. 
        # ok..... hm. So you if you itterate over npDC_2 thats the source node
        # you can make sub sets out of the pairs..... 
        clean_sent = CleanSpecialChars(sents_, ).split() # make sure this ends up as a list
        print('clean sentence', clean_sent)

        # Check to see if the nodes from the node list occur together in pairs within the sentence window
        clean_sent_word_hash = word_to_int(clean_sent)
        print('sentence word hash :', clean_sent_word_hash)
        for node in node_list:
            # so what you do is itterate through the source nodes
            # check it if is in the sentence
            if node in clean_sent:
                mask = node == pairs[:,0]
                sub_nodes = pairs[mask]
                sub_nodes_inds = np.where(mask)[0]
                print(node, sub_nodes)
                
                # Now you want a way to go through the second column and see what word is there
                # if they were numbers I'd multiply and look for perfect squares. 
                sub_nodes_hash = word_to_int(list(sub_nodes[:,1]))
                print(sub_nodes_hash)

                # use int 64
                opp = np.array(sub_nodes_hash, dtype=np.int64)[...,np.newaxis] * np.array(clean_sent_word_hash, dtype=np.int64)[np.newaxis,...]
                sqrt_opp = np.floor(np.sqrt(opp)).astype(np.int64)
                condition50 = sqrt_opp*sqrt_opp == opp
                sub_node_idx = np.where(condition50)[0]
                # check for perfect quare
                print(condition50)
                print(sub_node_idx)
                
                # Map back into global indices
                global_idx = sub_nodes_inds[sub_node_idx]

                # Increment counts
                pairs_counts[global_idx] += 1 / sentence_window
                # if you save the data frames over time..... then you get a full object that tells you how the relationships evolve over time.
                running_char_set.append(0+pairs_counts[...,np.newaxis])

    print("pair_counts :", pairs_counts)
    full_pairs_data= np.concatenate([pairs, pairs_counts[...,np.newaxis]], axis= -1)
    print(full_pairs_data)
    pd_full_pairs_data = pd.DataFrame(full_pairs_data, columns = ['x0','x1', 'counts'])
    pd_full_pairs_data['counts'] = pd_full_pairs_data['counts'].astype(float)
    pd_full_pairs_data = pd_full_pairs_data.sort_values(by="counts", ascending=False)           
    pd_full_pairs_data.to_csv(f'pair counts {sentence_window} sentence.csv')

    # concatenate for now
    running_char_set = np.concatenate(running_char_set, axis = -1)

    # get a top relationship 
    rcs_index = np.where(pairs[:,0]+pairs[:,1] == 'RaskolnikovPorfiry')[0][0]
    rcs_index2 =np.where(pairs[:,0]+pairs[:,1] == 'SoniaRaskolnikov')[0][0]
    print('rcs index : ', rcs_index)

    # this could be a function too. 
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(running_char_set[rcs_index, :], label = 'RaskolnikovPorfiry')
    axes[0].plot(running_char_set[rcs_index2, :], label = 'SoniaRaskolnikov')
    axes[0].set_ylabel('number of interactions')
    axes[0].set_xlabel('sentence number')
    axes[0].legend()

    axes[1].plot(np.diff(running_char_set[rcs_index, :]), label = 'RaskolnikovPorfiry')
    axes[1].plot(np.diff(running_char_set[rcs_index2, :]), label = 'SoniaRaskolnikov')
    axes[1].legend()
    
    plt.savefig(f'relationships_{sentence_window}sentence.png')

    return running_char_set, pd_full_pairs_data


if __name__ =='__main__':
    # Book = '/mnt/f/ebooks_public_domain/crime and punishment.txt'
    Book = '/mnt/f/ebooks_public_domain/crime and punishment.txt'
    text = GetText(Book)
    
    chars, special_chars = GetCharacters(text)
    print(list(special_chars))
    # print(list(text[0:10000]))

    # get all the paragraphs
    paragraph_inds, paragraphs = GetParagraphs(text)

    print(paragraphs[43].replace('\n', ' '))

    # get every sentence in every paragrph
    # the get sentences funciton will operate directly on the text, but at least this way we 
    # can link paragraphs to sentences
    sentence_starts, sentence_ends, sentences = GetSentencesFromParagraphs(paragraphs, paragraph_inds)

    for sent in sentences[100:150]:
        print('---')
        print(sent)

    # quotes are defined by special charcters in this text
    quoted_inds, quoted_texts = GetBoundText(text, 
                                            open_ = special_chars[-5], 
                                            close_ = special_chars[-4])
    for qtext in quoted_texts[100:150]:
        print('...')
        print(qtext)
    
    # get names of characters and what not. might hard code these. 
    DoubleCapped_starts, DoubleCapped_ends, DoubleCapped_ = GetDoubleCaps(text)
    print(sorted(list(set(DoubleCapped_))))

    # ------ new seciton 
    LikelyPlayers = GetLikelyPlayers(DoubleCapped_)
    
    # Get important pairs
    node_list, pairs, pairs_counts = GetImportantPairs(LikelyPlayers)
    
    # now after gettting a filtered list of word pairs, I want to get 
    # hits for when these 2 words occur in a given window. so like check sentence 500
    print('----- new section')
    
    # this thing 
    running_char_set, pd_full_pairs_data = DoSomething(node_list, pairs, sentences, sentence_window = 10)
    
    
   
    