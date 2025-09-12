import os 
import re
import numpy as np 
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

    # Get all the sentences
    sentence_starts = []
    sentence_ends = []
    sentences = []
    for para, pis in zip(paragraphs, paragraph_inds):
        a,b,c = GetSentences(para, offset = pis)
        sentence_starts.extend(a)
        sentence_ends.extend(b)
        sentences.extend(c)
    for sent in sentences[100:150]:
        print('---')
        print(sent)
    # quotes are defined by special charcters in this text
    quoted_inds, quoted_texts = GetBoundText(text, 
                                            open_=special_chars[-5], 
                                            close_ =special_chars[-4])
    for qtext in quoted_texts[100:150]:
        print('...')
        print(qtext)

    # get names of characters and what not. might hard code these. 
    DoubleCapped_starts, DoubleCapped_ends, DoubleCapped_ = GetDoubleCaps(text)
    print(sorted(list(set(DoubleCapped_))))

    # Get the individual words 
    DC_2 = []    
    for dc in DoubleCapped_:
        DC_2.extend(dc.split())
    
    DC_2_counts = GetWordCounts(DC_2)  
    sorted_items_descending = sorted(DC_2_counts.items(), key=lambda item: item[1], reverse=True)
    print(sorted_items_descending)
    

    # filter this so you dont get an unmanageable amount of word pairs
    thresh = 20
    common = ['Project','Gutenberg', 'The', 'But', 'And']
    top_DC_2 = {k:v for k,v in DC_2_counts.items() if v>thresh and not any([k== c for c in common])}  
    sorted_items_descending = sorted(top_DC_2.items(), key=lambda item: item[1], reverse=True)
    print(sorted_items_descending)
    
    # get word pairs 
    npDC_2 = np.array(list(top_DC_2.keys()))
    # print(npDC_2)
    n = len(npDC_2)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = i<j
    pairs = np.column_stack((npDC_2[i[mask]], npDC_2[j[mask]]))
    print(pairs.shape, n)
    print(pairs[0:-1])

    # now after gettting a filtered list of word pairs, I want to get 
    # hits for when these 2 words occur in a given window. 
    