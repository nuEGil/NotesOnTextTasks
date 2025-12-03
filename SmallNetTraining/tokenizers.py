import os 
import re
import numpy as np 
'''
Put all tokenizaiton options here.
Put all token prior computation here. 

Python ascii chart. Those x00 things are like start file and end file characters
https://python-reference.readthedocs.io/en/latest/docs/str/ASCII.html


Added some text normalization too. so gettext here is different then 
everywhere else now.... 
'''

def normalize_dashes(text):
    # Covers hyphen, en dash, em dash, figure dash, horizontal bar, etc.
    dash_pattern = r'[\u002D\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]'
    return re.sub(dash_pattern, '-', text)

def normalize_quotes(text):
    # Convert curly quotes, guillemets, and other Unicode doubles to plain "
    quote_pattern = r'[\u201C\u201D\u201E\u201F\u2033\u2036\u00AB\u00BB]'
    return re.sub(quote_pattern, '"', text)

def GetText(x):
    with open(x, "r", encoding='utf-8' ) as f:
        txt = f.read()
        txt = txt.replace('\t', '    ')
        txt = normalize_dashes(txt)
        txt = normalize_quotes(txt)
        return txt

def forbidden_ngram(x, depth=0, stop=2):
    if depth >= stop:
        return x
    # combine each element in x with every other element
    new_x = [x0 + x1 for x0 in x for x1 in x]
    return forbidden_ngram(new_x, depth + 1, stop)

def GetNgram(gram_tag = 'bigram'):    
    # You want 32 - 126 for this. 
    # charset = [[i,chr(i)] for i in range(32,126,1)]
    # print(charset)
    
    # basically everything on the key board - haha gpt has a higher vocab than I do. 
    charset = [chr(i)for i in range(32, 126, 1)] 
    charset.extend(['\n','\r'])
    print(''.join(charset))

    # ngram select
    gram_set = {
        'bigram' : lambda x : [x0+x1 for x0 in x for x1 in x],
        'trigram': lambda x : [x0+x1+x2 for x0 in x for x1 in x for x2 in x]}
    
    # alright make the gram from the character set
    ngram = sorted(gram_set[gram_tag](charset))
    print(f'first 100 {gram_tag}s')
    print(ngram[0:100], len(ngram), len(ngram[0]), len(charset))

    vocabulary_size = len(ngram) # needed 
    print('vocabulary size: ', vocabulary_size)

    # make encoders and make 
    # dictionary to convert between characters and indices 
    string_to_int = {ch : i for i, ch in enumerate(ngram)}
    int_to_string = {i : ch for i, ch in enumerate(ngram)}

    # make encoder and decoder functions
    def encode(s, char_len=len(ngram[0])):
        tokens = []
        for si in range(0, len(s)-char_len, char_len):
            substr = s[si:si+char_len]   
            # handle out of index tokens. we dont ever use _ so thats the blank       
            tokens.append(string_to_int.get(substr, string_to_int['.'*char_len]))
        return tokens
   
    def decode(x):
        return ''.join(int_to_string[i] for i in x) 

    # dump everything we might need
    outputs = {
        'vocabulary_size' : vocabulary_size,
        'encoder' : encode,
        'decoder' : decode,
        'string_to_int' : string_to_int,
        'int_to_string' : int_to_string,
        'char_len': len(ngram[0])
    }
    return outputs

def GenNgramProbs(gram_tag = 'bigram'):
    gram_set = {
        'bigram' : lambda x : [x0+x1 for x0 in x for x1 in x],
        'trigram': lambda x : [x0+x1+x2 for x0 in x for x1 in x for x2 in x]}
    
    # hardcoding the text files I want to use on this 
    text_dir = "/mnt/f/data/ebooks_public_domain/"
    # this happens to be the most common word list. 
    # Word list always ends with new line. likely to bias. 
    mc_text = GetText(os.path.join(text_dir, 'wordlist.10000.txt')) 
    mc_text = mc_text.replace('\n', ' ') # space is a more likely separator in books
    # tack on some more text from a book or something 
    mc_text2 = GetText(os.path.join(text_dir, 'The_Jungle.txt'))
    mc_text+=mc_text2
    # get unique characters 
    mc_chars = list(set(mc_text))
    # decide whether to filter out new line characters or not.
    # mc_chars = [mc for mc in mc_chars if not mc == '\n']

    # Three character strings 
    ngram_set = gram_set[gram_tag](mc_chars)
    char_len = len(ngram_set[0])
    # get the counts 
    counts = dict(zip(ngram_set, [0 for tt in ngram_set]))
    for ii, txt in enumerate(mc_text[0:-char_len]):
        if mc_text[ii:ii+char_len] in counts.keys():
            counts[mc_text[ii:ii+char_len]] +=1   
    
    # filter the counts because some values are 0  
    new_counts = {}    
    icounts = []
    sum_count = 0
    for k,v in counts.items():
        if v>0:
            new_counts[k]= 0+v
            sum_count+=v
            icounts.append([v,k])    
    
    icounts = sorted(icounts)[::-1]
    # new_probs = {k:v/sum_count for k,v in new_counts.items()}
    # iprobs = [[v,k] for k,v in new_probs.items()]
    # iprobs = sorted(iprobs)[::-1]

    print(f'char len:{len(mc_chars)}, total {char_len} char :{len(ngram_set)}')
    print(f'26**2:{26**2}, counts_len:{ len(counts)}')
    print(icounts[0:20])
    # print(iprobs[0:20])
    return new_counts

# only need to run this, and youll get back everything. 
def LoadVocabularySet(gram_='bigram'):
    chr_len = {
        'bigram'  : 2,
        'trigram' : 3}
    chr_len = chr_len[gram_]
    # get the main ngram tokenizers 
    gram_outputs = GetNgram(gram_tag = gram_)
    token_counts = GenNgramProbs(gram_tag = gram_)
    
    # Make a vector that is vocab len long, and has the probabilites
    # in the correct index
    token_probs = np.zeros((gram_outputs['vocabulary_size'],))
    for iii, (k_,v_) in enumerate(token_counts.items()):
        # print(k_)
        if k_ in gram_outputs['string_to_int'].keys():
            scoop = gram_outputs['string_to_int'][k_]
            token_probs[scoop] = 0+v_ 
        # else:
        #     print(f'token  {k_}  is outside of main set no prior needed')
    token_probs = token_probs/np.sum(token_probs)
    print('check priors : ', token_probs, np.sum(token_probs))
    
    gram_outputs['token_counts'] = token_counts
    gram_outputs['token_probs'] = token_probs
    return gram_outputs

if __name__ == '__main__':
    outputs = LoadVocabularySet(gram_='bigram')
    print(outputs.keys())
    