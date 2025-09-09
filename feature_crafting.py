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
'''

import re 

def GetText(x):
    with open(x, "r" ) as f:
        return f.read()
    
def GetWords(text, filter_chars = []):
    # get lines but join them by dots
    clean_text = '.'.join(text.splitlines())    
    # get rid of all the other characters
    clean_text = clean_text.replace(filter_chars[0], '.')
    for ch in filter_chars[1::]:
        clean_text = clean_text.replace(ch, '.')

    words = clean_text.split('.')
    words = sorted(list(set(words)))
    return words

def GetSentences(text,):
    # get lines but join them by space
    clean_text = ' '.join(text.splitlines())
    # sentence terminators
    # punctuations = ['...', '..', '. ', '!', '?']
    # put a space after the punctuation for it to be a line terminator
    punctuations = ['. ', '! ', '? ']
    # use regext to get all the indices in a dicitonary 
    indices = {p: [m.start() for m in re.finditer(re.escape(p), clean_text)] for p in punctuations}

    # put all the indices together though so we can grab any sentence that we want. 
    full_inds = [0]
    for k,v in indices.items():
        full_inds.extend(v)
    full_inds  = sorted(full_inds)
    
    # get number of sentences
    n_sentences = len(full_inds) - 1 #start stop, then next sentence is prevs stop

    print('min full inds ', min(full_inds))
    print('max full inds ', max(full_inds))
    print('number of sentences : ', n_sentences)
    
    sentences = [ ]
    for sentence_id in range(0,n_sentences):
        # sentence_id = n_sentences-1
        # sentence_id = 190
        sentence = clean_text[full_inds[sentence_id]+2:full_inds[sentence_id+1]+1]
        sentences.append(sentence)
        # print('--'+sentence)

    # print(sentence.split())
    # {'.': [27, 46], '!': [11], '?': [41]}

    # print(clean_text)
    return sentences

def GetNouns(sentences):
    # if you want to get full names you might have to get a regular expression... 
    # itterate through sentences
    nouns = []
    for sentence in sentences:
        words = sentence.split()
        # print(words)
        for wi, w in enumerate(words):
            w = ''.join([w_ for w_ in w if w_.isalnum()])
            if len(w)>3:
                if wi>=1 and w[0].isupper() and not w[1].isupper():
                    nouns.append(w)
    print(sorted(list(set(nouns))))
    return 0

if __name__ == '__main__':
    Book = '/mnt/f/ebooks_public_domain/crime and punishment.txt'
    english_words = '/mnt/f/ebooks_public_domain/wordlist.10000.txt'
    english_words = GetText(english_words)
    # print(english_words.splitlines())
    
    # # Book = '/mnt/f/ebooks_public_domain/The jungle by upton sinclair.txt'
    # # Book = '/mnt/f/ebooks_public_domain/Federalist papers.txt'
    text = GetText(Book)
    
    chars = sorted(list(set(text))) # get unique characters
    alpha_numeric_chars = [c for c in chars if c.isalnum()]
    other_chars = list(set(chars) - set(alpha_numeric_chars))
    
    # allwords = GetWords(text, filter_chars = other_chars)
    # GetNouns(text, filter_chars=other_chars)

    print('Full Char set', chars)
    print('alpha numeric set', alpha_numeric_chars)
    print('other', other_chars)
    # print('vocabulary size ', len(allwords))

    sentences = GetSentences(text,)
    GetNouns(sentences)
    

    
    