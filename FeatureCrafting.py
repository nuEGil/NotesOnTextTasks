'''
Crafting some text features . 
In books any way, we know that sentences prety much always end in .? or ! right like termination characters 
we know  ;,: these are used to change direction in the thought. 

But I also care about pulling features like the number of characters per sentence right. 

I want to be able to get a few common phrases too. that express interesting things

Transition phrases
https://www.vanderbilt.edu/writing/resources/handouts/transitions/?utm_source=chatgpt.com

Punctutaion 
https://www.thepunctuationguide.com/?utm_source=chatgpt.com

So, we can make a crude text segmentation tool, by understanding transition phrases, and punctuations

You have to be careful with some of this stuff in other texts though. 
Because soon in the wealth of nations and the federalist papers could be like 

we will do this soon 

or I would sooner be drowned than spend another minute here. 

So that's a whole thing. 

--------
Huge note. Humans build up knowledge of the book as we read.
We read one line at a time, or jump ahead, but we are looking for 
Who, What, Where, When, Why.

When we talk we are saying the same thing... 

So maybe that's something to think about . 

The other thing is that the line numbers tell you something too. 

The following numbers should mean something. just not sure what. 
Moving average to get signals 
number of  / number of 
chars / words
chars / sentence
chars / paragraph
words / sentence
words / paragraph
sentence / paragraph

Vocabulary size after each sentence 
because you might use a new word when you introduce a new idea.

Could you Identify  key points in the text and just pass in those 
passages to the network? Context doesnt have to be the whole file. 

--- 
Also, by no means are we going for efficiency with this code. 

'''

import re 
import argparse
import numpy as np 

def get_args():
    parser = argparse.ArgumentParser(description="Extract image patches with stride.")
    parser.add_argument("--file", type=str, required=False, help="Path to the input file")
    return parser.parse_args()


# This is probably a class right here. ----
def GetText(x):
    with open(x, "r" ) as f:
        return f.read()

def GetParagraphs(text):
    # I happen to want the indexes, but you can get a regular expression to just grab paragraphs. 
    paragraph_inds = [0]
    paragraph_inds.extend([m.start() for m in re.finditer(r"\n\n", text)])
    n_paragraphs = len(paragraph_inds) - 1
    
    paragraphs = []
    for ii in range(n_paragraphs):
        sub = text[paragraph_inds[ii]+2 : paragraph_inds[ii+1]]
        sub = ' '.join(sub.splitlines())
        paragraphs.append(sub)

        # print a few
        if ii<50:
            print(f'paragraph_{ii}: ', sub)
            print('==')

    return paragraphs, paragraph_inds

def GetSentences(text, punctuations = [': ', '. ', '! ', '? ']):
    # use regext to get all the indices in a dicitonary 
    indices = {p: [m.start() for m in re.finditer(re.escape(p), text)] for p in punctuations}
    
    # put all the indices together though so we can grab any sentence that we want. 
    sentence_inds = [0]
    for k,v in indices.items():
        sentence_inds.extend(v)
    sentence_inds  = sorted(sentence_inds)
    n_sentences = len(sentence_inds) - 1 #start stop, then next sentence is prevs stop
    
    # ok now build an object that is all the sentences
    sentences = [ ]
    for sentence_id in range(0, n_sentences):
        sentence = text[sentence_inds[sentence_id]+1:sentence_inds[sentence_id+1]+1]
        sentences.append(sentence)
    return sentences, sentence_inds

def GetCharacters(text):
    # get Unique characters 
    chars = sorted(list(set(text)))
    letters = sorted([c for c in chars  if c.isalpha()]) 
    special_chars = "".join(list(set(chars) - set(letters)))
    
    print(chars)
    print(special_chars)
    return chars, special_chars

def GetWords(text, special_chars = '!?.'):
    # get all unique words
    pattern = "[" + re.escape(special_chars) + "]"
    f_text = re.sub(pattern, " ", text)
    words = f_text.split()
    # words  = [''.join([w_ for w_ in wor.lower() if w_.isalpha()]) for wor in f_text]
    words = [w_.lower() for w_ in words]
    words = sorted(list(set(words)))
    print(words)
    return words

def GetWordsAndInds(text, special_chars = '.'):
    pattern = "[" + re.escape(special_chars) + "]"
    # replace the special character with a space
    f_text = re.sub(pattern, " ", text)
    
    # now you have every word in order of occurence 
    words = f_text.split()
    
    uwords = set()
    uwords_and_locs = []
    for wi, word_ in enumerate(words):
        # now you have an index and each word
        # I want to know that it is alpha numeric
        filtered_word = ''.join([fw_ for fw_ in word_.lower() if fw_.isalpha()])
        if not filtered_word in uwords:
            uwords.add(filtered_word)        
            uwords_and_locs.append([filtered_word,wi])
    return uwords_and_locs
# end of the class right here -- these can group together. 

def GetNounsFromSentence(text_):
    # sentence is already spaced out text 
    words_ = text_.split()
    nouns = []
    for wi, w in enumerate(words_):
        if len(w)>1:
            if w[0].isupper() and not w[1].isupper() and wi>1:
                w = ''.join([ww for ww in w if ww.isalnum()])
                nouns.append(w)
    return nouns

def FirstRead(Book):
    text = GetText(Book)
    # get all the paragraphs
    paragraphs, paragraph_inds = GetParagraphs(text)
    
    # get all the sentences
    # filter the text so that it is easier to get sentences
    s_text = text.replace('\n\n', ' ')
    s_text = s_text.replace('\n', ' ')
    sentences, sentence_inds = GetSentences(s_text)
    
    for si, sent in enumerate(sentences):
        print(f'sentence {si} : ', sent)

    # get Nouns from sentences:
    all_nouns = []
    for si, sent in enumerate(sentences):
        nouns = GetNounsFromSentence(sent)
        all_nouns.extend(nouns)
    all_nouns = sorted(list(set(all_nouns)))
    print(all_nouns)

    # get characters 
    chars, special_chars = GetCharacters(text)

    # words = GetWords(text, special_chars = special_chars)
    uwords_and_locs = GetWordsAndInds(text, special_chars = special_chars)

    print('Total unique chars : ', len(chars))
    # print('Total unique words : ', len(words))
    print('Total U words : ', len(uwords_and_locs))
    print('Total sentences : ', len(sentences))
    print('Total paragraphs : ', len(paragraphs))

    # might just bite the bullet and use gpt2 or gemma3 to generate some signals.
    
    # Here's the inefficiency bit, there's 3 copies of the text here... 
    # could probably fix that by just storing indices for paragraphs and sentences
    # words is the only one that removes characters.... 
     
    data = {
        'text':text,
        'paragraphs' : paragraphs,
        'paragraph_inds' : paragraph_inds,
        'sentences' : sentences,
        'sentence_inds' : sentence_inds,
        'words_and_inds': uwords_and_locs,
        'nouns' : nouns,
        'chars' : chars,
        'special_chars' : special_chars,
        'Tn_unique_chars': len(chars),
        'Tn_words' : len(uwords_and_locs),
        'Tn_paragraphs': len(paragraphs)
        }
    
    return data


if __name__ == '__main__':
    xargs = get_args()
    Book = '/mnt/f/ebooks_public_domain/crime and punishment.txt'
    # Book = xargs.file 
    book_dat = FirstRead(Book)

    # ok so for the next part. I want to calculate a bunch of features from portions of the text. 
    
    # you could count how many sentences were in a given paragraph -- you need an index filter though
    print(book_dat['paragraph_inds'][0:2])
    print(book_dat['sentence_inds'][0:10])

    # ok so now if I want to get the character index where the new thing occured
    