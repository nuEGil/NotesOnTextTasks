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

Podcast analyzer is going to be different -- we have different speakers. 
So you need to grab the text by the time stamps. 

so these sets of tools.... likely you make a class for processing text in 
project gutten berg books... 

and then another class for working with lex podcasts. 
especially because the Lex Stuff is real conversations.. 


Implement a guttenberg filter. 
'''

import re 
import argparse
import scipy 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from collections import Counter

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

    print('characters : ', chars)
    print('special characters : ', special_chars)

    # words = GetWords(text, special_chars = special_chars)
    uwords_and_locs, total_n_words = GetWordsAndInds(text, special_chars = special_chars)

    print('Total Book characters : ', len(text))
    print('Total unique chars : ', len(chars))
    # print('Total unique words : ', len(words))
    print('Total number of words : ', total_n_words)
    print('Total number of unique words : ', len(uwords_and_locs))
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
        'Tn_chars': len(text),
        'Tn_unique_chars': len(chars),
        'Tn_words' : total_n_words,
        'Tn_uwords' : len(uwords_and_locs),
        'Tn_paragraphs': len(paragraphs)
        }
    
    return data

def plot_word_signals(book_dat):
    
    ## Word Signals
    # ok so now if I want to get the character index where the new thing occured
    new_word_inds = np.array([bd[1] for bd in book_dat['words_and_inds']])
    word_signal = np.ones((book_dat['Tn_words'],))
    for ind_ in new_word_inds:
        word_signal[ind_::]+=1
    
    # this feature is a bit more expensive to capture. 
    filt_word_signal = scipy.signal.savgol_filter(word_signal, 31, 2)
    residual_word_sig = word_signal-filt_word_signal
    # it does spike up and down a lot, and the 0 crossings should tell you something. but idk yet 
    # plotting
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0,0].plot(new_word_inds)
    axes[0,0].set_ylabel('index a new word was introduced')
    axes[0,0].set_xlabel('word id')
    
    # this interval will be longer toward the end -- you care about spikes
    # from the base line -- it should mean something when the inteval spikes or drops 
    # for some reason -- that difference should give you like a normalized version of the 
    # interval between words signal. More words should mean the speaker is introducing a new concept
    # less words means the speaker is either elaborating or in a lull. 


    axes[0,1].plot(np.diff(new_word_inds))
    axes[0,1].set_ylabel('interval between new words')
    axes[0,1].set_xlabel('word id')
    
    axes[1,0].plot(word_signal)
    axes[1,0].plot(filt_word_signal)

    axes[1,0].set_ylabel('number of unique words')
    
    # # this should be 1s and 0s so not meaningful... 
    # axes[1,1].plot(np.diff(word_signal))
    # axes[1,1].set_ylabel('diff number of unique words')

    # this should be 1s and 0s so not meaningful... 
    # axes[1,1].plot(np.diff(new_word_inds) / book_dat['Tn_words'])
    axes[1,1].plot(residual_word_sig)
    axes[1,1].set_ylabel('residual word sig')
    axes[1,1].set_xlabel('word id')
    plt.savefig('word_signals.png')

if __name__ == '__main__':
    xargs = get_args()
    Book = '/mnt/f/ebooks_public_domain/crime and punishment.txt'
    # Book = '/mnt/f/podcast/Lex Fridman Transcript for Keyu Jin Chinas Eco.txt'
    # Book = '/mnt/f/ebooks_public_domain/Time machine.txt'
    # Book = xargs.file 
    book_dat = FirstRead(Book)

    # ok so for the next part. I want to calculate a bunch of features from portions of the text. 
    
    # you could count how many sentences were in a given paragraph -- you need an index filter though
    print(book_dat['paragraph_inds'][0:2])
    print(book_dat['sentence_inds'][0:10])

    # puncuation variability... over a window.... characters or paragraphs... 
    paragraph_mean_spchar = []
    paragraph_std_spchar = []
    for paragraph in book_dat['paragraphs']:

        counts = dict(zip(list(book_dat['special_chars']),
                          [0]*len(book_dat['special_chars'])))
        for ch in paragraph:
            if ch in counts.keys():
                counts[ch]+=1

        count_dat = pd.DataFrame.from_dict({'character':counts.keys(), 'count':counts.values()})
        print(paragraph)
        # print(count_dat.sort_values(by=count_dat.columns[1]))
        # now count dat is dropped if it is 0
        count_dat = count_dat.loc[(count_dat != 0).any(axis=1)]
        pmean = count_dat.iloc[:,1].mean()
        pstd = count_dat.iloc[:,1].std()
        print('average special character count ', pmean)
        print('std special character count ', pstd)

        paragraph_mean_spchar.append(pmean)
        paragraph_std_spchar.append(pstd)
    
    new_paragraph_data = {
        'paragraph':book_dat['paragraphs'],
        'avg_special_char': paragraph_mean_spchar,
        'std_special_char': paragraph_std_spchar
        }
    pd.DataFrame.from_dict(new_paragraph_data).to_csv(Book.replace('.txt', '_pargraph_feats.csv'))


    ### end of feature computation 

    paragraph_mean_spchar = np.array(paragraph_mean_spchar)
    paragraph_std_spchar = np.array(paragraph_std_spchar)
    
    rms_mean = np.sqrt(np.mean(paragraph_mean_spchar))
    rms_std = np.sqrt(np.mean(paragraph_std_spchar))
    print('rms of paragraph mean num chars ', rms_mean)
    print('rms of paragraph std num chars ', rms_std)
    
    
    both_conditions = np.where(
    (paragraph_mean_spchar > 5 * rms_mean) & 
    (paragraph_std_spchar > 2.5 * rms_std))[0]


    print("both conditions:", both_conditions)
    print("count:", len(both_conditions))
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    axes[0].plot(paragraph_mean_spchar)
    axes[0].plot(paragraph_mean_spchar+paragraph_std_spchar)
    axes[0].plot(paragraph_mean_spchar-paragraph_std_spchar)
    axes[1].plot(paragraph_mean_spchar)
    axes[1].axhline(y = 5*rms_mean)
    axes[1].axhline(y = 2.5*rms_std)

    plt.savefig('punct_var.png')




    