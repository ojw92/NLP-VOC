

# preprocessor.py

# Preprocessing steps for RNN & CNN

# Model for S22 VOC only, but with added preprocessing steps

# https://www.geeksforgeeks.org/python-call-function-from-another-file/

from timeit import default_timer as timer

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')    # for lemmatization function; don't need this line if package installed
nltk.download('omw-1.4')    # for lemmatization function; don't need this line if package installed
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer     # based on The Porter Stemming Algorithm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91

# 0 - split & prepare data for preprocessing                              <- add a preliminary stopword removal step (RNN & CNN)?
# 0.5 - remove duplicate data
# 1 - remove hyperlinks, emails & pricetags
# 2 - remove emoji
# 3 - remove punctuations
# 4 - apply lower case

# ===== use min-df to find low-frequency words =====

# 5 - fix misspellings                                                    <- (need to implement!)
# 6 - replace words with generic nouns                                    <- (need to implement!)
# 7 - remove stopwords       # better to replace then remove?             <- final stopword clean-up step (separate list for RNN & CNN)
# 8 - lemmatize
# (8.5 - stemming)

# 9 - tokenization & padding

# ============================================================================================================= 0                 <- add a preliminary stopword removal step

def data_split(df22):

    # Split the input dataframe into Text (Title + Contents) and Classes dataframes
    # input must have 3 columns of string entries (Title, Contents, and Classes)

    
    # print(df22.describe())     # get an overview of the dataset

    # check if every row entry of each column is string type (some are NaN, so no)
    # print(df22.applymap(lambda x : type(x).__name__).eq({'Title': 'str', 'Content': 'str', 'Class':'str'}))

    # convert NaN to empty strings (NaN -> str)
    df22 = df22.replace(np.nan, '', regex=True)

    # concatenate strings of title & content with a " " in between (1 body of text)
    text22 = df22.iloc[:,0] + " " + df22.iloc[:,1]      # slicing DataFrame makes it a Series
    text22 = pd.DataFrame(text22, columns= ['Text'])    # so initialize it as a DataFrame. pd.DataFrame(some_Series) works
    classes22 = df22.iloc[:,2]

    classes22 = classes22.replace(to_replace="R", value=1)
    classes22 = classes22.replace(to_replace="r", value=1)
    classes22 = classes22.replace(to_replace="N", value=0)
    classes22 = classes22.replace(to_replace="n", value=0)
    classes22 = pd.DataFrame(classes22, columns=['Class']).astype('int32')
    # classes22.columns = ['Class']    # classes22=pd.DataFrame(classes22) causes an error cus   pd.DataFrame(some_DataFrame) makes no sense - should pass a list

    #print(text22.head(10))
    #print(classes22.head(10))
    print('==================================================')

    return text22, classes22



# ============================================================================================================= 0.5

def drop_dupe_text(text22, classes22):

    # df22.drop_duplicates() will remove extra rows that have the same values in all columns (redundant)
    # As a result only duplicate rows left are pairs of datapoints with 0 & 1 as their labels
    # This function removes all of rows from those pairs with 0 as their label, as they can be disruptive
    # in training the model. Label 1 should take priority

    textdupeindex = text22[text22.duplicated(keep=False)==True].index
    dupeclasses = classes22.filter(items = textdupeindex, axis=0)
    redundantclass0 = dupeclasses[dupeclasses.Class==0]
    text22.drop(index=redundantclass0.index, inplace=True)
    classes22.drop(index=redundantclass0.index, inplace=True)

    return text22, classes22



# ============================================================================================================= 1

def hep_remove(text22):

    # Remove hyperlinks, e-mails, and pricetags from a dataframe of strings
    ### It would be nice to instead of removing them, replace e-mails and hyperlinks with "url"
    ### and replace price tags with "price"

    # algorithm
    # regex that matches all non-whitespace text before and after '.com', '.org', '.gov', '.edu'
    # "" before and after '@', but the above line should take care of this automatically
    # "" after '$£€₱₽¥₩'; number, comma, and period should suffice, but account for pricetags written in text

    # list of hyperlinks, e-mails, and price tags
    # use this step specifically to remove words that have punctuation mixed in
    heplist = ['.com','.edu','.org','.gov', '.co', 'https:', 'http:',
                '$','£','€','₱','₽','¥','₩', ',000', '.00']   # include '-'? need to think
    # remove hyperlinks (& e-mails, as side effect!)
    text22 = [' '.join(y for y in x.split() if not any(ele in y for ele in heplist)) 
                for x in text22.Text]
    text22 = pd.DataFrame(text22, columns= ['Text'])

    return text22



# ============================================================================================================= 2

def emoji_remove(text22):

    # Remove image & textual imoji

    # Remove emoji
    # https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d
    def emoji_rem(string):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)
    
    text22.Text = text22['Text'].apply(emoji_rem)
    # text22 = list(map(emoji_rem, text22))     # for a list, map the function

    """
    # Remove texual emoji and other colon  (":fire:", ":hundred_points",)
    def textualemoji(string):
        return re.sub(r'[^:\w:$]', ' ', string)     # starts & ends with :, matches text, number and _ between :'s

    text22.Text = text22['Text'].apply(textualemoji)
    """
    # Too many 'letter emojis' - need to use regex
    """letteremoji = ['Q_Q','QQ','T_T','UWU','uwu','UwU','uWu', 'orz','OTL',
                   'o_o','O_O','o_O','O_o', 'o-o','O-O', '==','=_=','-_-','--',
                   '*_*','$_$','@_@','?_?','+_+','>_>','<_<', '~_~', "'-'",
                   ':x',';x', ':D',';D','D:','D;', ':)',';)', ':(',';(', '(:','(;','):',');',
                   ':s',':S']"""
    
    #text22 = [' '.join(y for y in x.split() if not any(ele in y for ele in letteremoji)) 
    #            for x in text22]
    # text22 = pd.DataFrame(text22, columns=['Text'])
    
    return text22

# ===================== Implement regex in letteremoji later ========================



# ============================================================================================================= 3

def punc_remove(text22):
    
    # Remove punctuation
    
    # using RegexpTokenizer might lose words like "Mr."
    # need to remove just the punctuation, not the whole word attached to it!
    # what about words like, 'work-life-balance' ?

    # double backslash for a string of single backslash
    # punclist = ['!','@','#','$','%','^','&','*','(',')','-','_','=','+','\\','|','`','~',',','.','<','>','/','?',';',':','"',"'"]


    def puncrem(string):
        return re.sub(r'[^\w\s]', '', string)

    text22 = text22['Text'].apply(puncrem)
    text22 = pd.DataFrame(text22, columns= ['Text'])

    return text22

# If we get rid of punctuation in things like "it's" or ":S", we're gonna get
# single letter words or misspelled words (it's -> its)
# for CNN, we can just remove them, but RNN we need a fix...



# ============================================================================================================= 4

def lower_case(text22):

    # Take a data frame of strings and make them lower case characters
    
    # First replace any NaNs generated from other preprocessing steps with empty strings
    text22 = text22.replace(np.nan, '', regex=True)

    # Apply lower case
    text22 = text22['Text'].str.lower()                 # Series
    text22 = pd.DataFrame(text22, columns= ['Text'])

    return text22



# ============================================================================================================= 5               <- (need to implement!)

def fix_spelling(text22):
    
    # Take a data frame of strings and fix all misspelled words

    
    # logic here



    text22 = pd.DataFrame(text22, columns= ['Text'])

    return text22




# ============================================================================================================= 6               <- (need to implement!)

def pn_replace(text22):

    # Take a data frame of strings and replace proper nouns or certain nouns with a common, generic noun

    # Texas, New York City, NYC, LA, Los Angeles, NJ --> usloc
    # China, Hong Kong, New Zealand, UK, Canada, India --> nonusloc
    # Max, Mark, Johnny, Adam, Obama, Kanye --> person
    # bees, duck, my dog, dog, cat, falcon, eagle --> animal      (for a stopword list, make sure "my dog" appears before "dog"
    # he, she, their, us, it pronouns --> pronoun
    # use this for ideas https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    # exynos vs snapdragon

    #querywords = query.split()
    #print(querywords)
    #resultwords  = [word for word in querywords if word.lower() not in stopwords]
    #result = ' '.join(resultwords)


    text22 = pd.DataFrame(text22, columns= ['Text'])

    return text22
    


# ============================================================================================================= 7

# Stopword will need a lot of work! Expect performance to be bad/worse with stopwords
# For the low-freq words the stopwords can't catch, use min-df to clean up

def stopword_remove(text22, model="RNN"):

    # Take a data frame of strings and remove all preselected stopwords from them

    # Remove stopwords (including my choice of words)
    # may need to keep words like "not", "doesn't", "does"
    #from nltk.corpus import stopwords
    #nltk.download('stopwords')     # don't need this line if package installed
    #stop = stopwords.words('english')
    """i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, yourself,
    yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself, they, them, their,
    theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am, is, are, was, were, be, been, 
    being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of,
    at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down,
    in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each,
    few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don,
    don't, should, should've, now, d, ll, m, o, re, ve, y, ain, aren, aren't, couldn, couldn't, didn, didn't, doesn,
    doesn't, hadn, hadn't, hasn, hasn't, haven, haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn,
    needn't, shan, shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't 
    """
    # lots of words in nltk stopwords list have significance in VOC context, so make my own list:
    # if using CNN, use stopwords. If using LSTM(RNN), may need to think about it. lemmatizing LSTM can get tricky too
    """
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'will', 'just',
            "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 
            'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
            'weren', "weren't"]
    """
    # use this basic list for now - study data closely and add more later
    if model == "RNN":
        # to/from, up/down, in/out opposite meanings -> significant? "be" verb not significant
        # who/what/when/where/why/how, which, only
        stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if',
            'or', 'as', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between',
            'into', 'through', 'during',
            'to', 'from', 'in', 'out',     # to/from, in/out,      on/off, up/down, above/below, over/under
            'again', 'further', 'then', 'once', 'here', 'there',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'will', 'just',
            'd', 'll', 'm', 'o', 're', 've', 'y', 'ma']
    elif model == "CNN":
        stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between',
            'into', 'through', 'during', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'will', 'just',
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'ma']
    
    # create my own list of stopwords and concatenate to existing list of 179 NLTK stopwords
    my_stopwords = ['lol', 'lmao', 'lmfao', 'rofl', 'wtf','fuck','shit','jesus','christ','wth','what the fuck',
                    'what the hell','samsung','apple','lg','xiaomi',
                    'cus','cuz',
                    '0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                    'hi','hey','hello','hiya','greetings','all','sincerely','bye','farewell',
                    'thanks','thank','appreciate','think','believe',
                    'probably','almost','likely', 'sometimes','frequently','occasionally','gradually','occasion','gradual',
                    'expensive','cheap','price','pricey',
                    'really'
                    ]
    # concatenate RNN/CNN stopwords with my stopwords
    stop = [*stop, *my_stopwords]

    #text22 = text22.apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)])) # joins all rows for some reason
    removeit = np.vectorize(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
    text22 = pd.DataFrame(removeit(text22), columns= ['Text'])

    return text22



# ============================================================================================================= 8

# Lemmatization - 'caring' --> 'care'

def lemmatize_df(text22):

    # Take a data frame of strings and lemmatize its texts
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    wnl = WordNetLemmatizer()   # plural --> singular, but verb tense unchanged

    # This is my own, long (high-CPU-usage?) way to tokenize & lemmatize words!
    tempdf = pd.DataFrame(columns=['Text'])
    for i in text22['Text']:
        jol = []
        for j in w_tokenizer.tokenize(i):
            jol.append(wnl.lemmatize(j))
        tempdf.loc[len(tempdf.index)] = [jol]
        # text22['Text'].loc[len(text22.index)] = [jol]   # find a way to append directly to existing df and trim it
    # print(tempdf.head(10))
    # print("length of tempdf: " , len(tempdf))

    text22 = pd.DataFrame(tempdf, columns=['Text'])

    return text22

# ===== A much shorter way to lemmatize! (more CPU-efficient?) =====
"""
def lemmatize_text(some_string):
    return [wnl.lemmatize(w) for w in w_tokenizer.tokenize(some_string)]
df22['cont_sw_lem'] = df22['cont_sw'].apply(lemmatize_text)    # .apply works on DataFrame or Series
print('\nAfter lower case & lemmatization: \n')
print(df22.head())
"""



# ============================================================================================================= 8.5

# Stemming - 'caring' --> 'car'
# No need to use both lemmatization & stemming - just use lemmatization

def stemming_df(text22):

    # Take a data frame of strings and do stemming on its texts
    
    # nature, natures, natural --> natur
    snowball_stemmer = SnowballStemmer('english')
    # kol = []     # This is my way of stemming on one Series
    # for i in text22['Text']:
    #     kol.append([snowball_stemmer.stem(word) for word in i])
    # stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
    # text22['Text'] = kol
    
    def stemit(some_string):
        return [snowball_stemmer.stem(word) for word in some_string]
    
    text22 = text22.Text.apply(stemit)
    text22 = pd.DataFrame(text22, columns= ['Text'])

    # text22['Text'] = text22.Text.apply(stemit)

    #print(text22['Text'][2715])

    return text22






# ============================================================================================================= 9


# Word tokenization

def tokenize_pad_words(text22, min_df=0, max_xref=0):
    
    # Take a data frame of strings to tokenize its words, then assign an index to each word

    #      tokenization
    # =========================
    # https://www.kaggle.com/code/hamishdickson/using-keras-oov-tokens
    # https://stackoverflow.com/questions/49073673/include-punctuation-in-keras-tokenizer
        # should I include punctuations? Only like 30 more features...
    token22 = Tokenizer(num_words=11000, lower=True, oov_token='OOV')  #20221012 data 10920 words w freq>2
    token22.fit_on_texts(text22['Text'])
    print('\nIndex determined by word freq(hi->lo) : \n', token22.word_index)    # Dict object - index determined by word frequency highest-lowest

    # Convert each voc entry to list of tokenized words, then find longest list (most words in entry)
    x22 = token22.texts_to_sequences(text22['Text'])
    max_x22 = max(len(elem) for elem in x22)
    print('\nVOC entries represented as tokenized indices : \n', x22[0:2])      # List of lists - too long to print in full
    print('\nFrequency of each word : \n', token22.word_counts)                 # Dict
    print('\nNumber of sentences each word appears in : \n', sorted(token22.word_docs.items(), key=lambda z: z[1], reverse=True))  # [('the', 3632), ('i', 3010), ('to', 2989), ...]
        # sorted(token22.word_docs.items())    [('0': 17), ('00': 1), ('000a': 1), ...] 
    print('\nTotal number of VOC : \n', token22.document_count)
    print('\nLongest VOC entry has %d (tokenized) words.' % max_x22)

    word_size22 = len(token22.word_index) + 1


    # Take a list of lists of tokenized words and do padding on them

    #      padding
    # =========================
    # each VOC has different number of words, so x has a list of lists, which is why to_categorical(x) caused error
    # normalize every voc to vectors of same length - max_x
    if max_x22 >= max_xref:
        max_x22 = max_x22
    else:
        max_x22 = max_xref
    padded_x22 = pad_sequences(x22 , max_x22)
    print('\nPadded results : \n', padded_x22)
    print('type of padded_x22 is array of lists/matrix?  ', type(padded_x22))            # <---------- delete this later after finding out what it is
    print('Length of padded_x22: ', len(padded_x22))
    print('word_size22: ', word_size22, ' (number of unique words + 1)')
    print('max_x22: ', max_x22)


    # Feature to add later - minimum word frequency to simplify our tokenized list
    # Simple line below can filter by word count > 1. But need to apply to padded_x22 & max_x22, too
        # For now, use Tokenizer(num_words=11000) since 20221012 data has 10920 words with frequency > 2
    # mindf1 = {k:v for (k,v) in token22.word_counts.items() if v > 1}
    # print(len(mindf1))


    # print(padded_x22.dtype)    # int32
    # print(classes22.dtype)    # int64 


    return padded_x22, word_size22, max_x22



# ============================================================================================================= 9


# Visualizing Word2Vec Embeddings with t-SNE
import gensim
from sklearn.manifold import TSNE
import random

def semantic_similarity(X_train):

    # X_train should be a single-column DataFrame of preprocessed Title + Content (text22 or df22['Text'])
    X_train = X_train.apply(lambda x: gensim.utils.simple_preprocess(str(x)))
    # Train the word2vec model
    w2v_model = gensim.models.Word2Vec(X_train,
                                    vector_size=100,   # size of vectors desired
                                    window=5,  # # of words before & after target word to use as context
                                    min_count=2        # min_df
                                    )
    w2v_model.build_vocab(X_train)  # prepare the model vocabulary
    w2v_model.train(X_train, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
    vocab_size, embedding_size = w2v_model.wv.vectors.shape

    n_samples = 250
    # Sample random words from model dictionary
    random_i = random.sample(range(vocab_size), n_samples)
    # def token2word(token):
    #     return w2v_model.wv.index_to_key[token]
    random_w = [w2v_model.wv.index_to_key[i] for i in random_i]

    # Generate Word2Vec embeddings of each word
    word_vecs = np.array([w2v_model.wv[w] for w in random_w])   # 'Word2Vec' object not subscriptable; use .wv

    # Apply t-SNE to Word2Vec embeddings, reducing to 2 dims
    tsne = TSNE()
    tsne_e = tsne.fit_transform(word_vecs)

    # Plot t-SNE result
    plt.figure(figsize=(32, 32))
    plt.scatter(tsne_e[:, 0], tsne_e[:, 1], marker='o', c=range(len(random_w)), cmap=plt.get_cmap('Spectral'))

    for label, x, y, in zip(random_w, tsne_e[:, 0], tsne_e[:, 1]):
        plt.annotate(label,
                    xy=(x, y), xytext=(0, 15),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.1))


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
def scatterplot_VOC(x, y, number_of_R, threshold):
    # x = array of len(y)                      y = array of prediction probabilities
    # number_of_R = number of actual R + 1     threshold = prediction threshold in float
    plt.scatter(x, y_prob3, c=(np.where(x<number_of_R+1,'g', 'r')))      # plots VOC (R & N)
    plt.plot([0, len(y)],[threshold, threshold], c='k', linestyle='--')  # plots threshold

    ###handles, labels = plt.gca().get_legend_handles_labels()    # *use this to add new to existing handle
    R_points = mpatches.Patch(color='g', label='R')
    N_points = mpatches.Patch(color='r', label='N')
    threshline = Line2D([0], [0], color='k', linestyle='--', label='threshold')
    ###handles.extend([R_points, N_points, threshline])           # *also plt.legend(handles=handles)

    plt.legend(handles=[R_points, N_points, threshline], loc="best")
    plt.xlabel('VOC #')
    plt.ylabel('Predictions')



