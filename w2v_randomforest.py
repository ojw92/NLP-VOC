# reference: https://medium.com/@dilip.voleti/classification-using-word2vec-b1d79d375381
# ver. 20221025

# Read in the data and clean up column names
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)

df22_1 = pd.read_csv('20221011pre1.csv', sep='\t')    # NA at 2736
df22_2 = pd.read_csv('20221011pre2.csv', sep='\t') 
df22_3 = pd.read_csv('20221011pre3.csv', sep='\t')    # NA at 427
df22_4 = pd.read_csv('20221011pre4.csv', sep='\t')    # Class column has 'N' and 'n'

df22_1.Class.iloc[2736] = df22_1.Content.iloc[2736]
df22_1.Content.iloc[2736] = ''
df22_3.Class.iloc[427] = df22_3.Content.iloc[427]
df22_3.Content.iloc[427] = ''
df22 = pd.concat([df22_1, df22_2, df22_3, df22_4], ignore_index=True).drop_duplicates()
# df22 = df22.sample(frac=1, random_state=seed)

# Clean data using the built-in cleaner in gensim (lowercase, remove all punctuation (it's->it) & tokenize)
df22['Text'] = df22.iloc[:,0] + " " + df22.iloc[:,1]
df22['Text'] = df22['Text'].apply(lambda x: gensim.utils.simple_preprocess(str(x)))

# Encoding the label column
df22['Class'] = df22['Class'].map({'R':1,'N':0,'n':0})
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df22['Text'], df22['Class'] , test_size=0.2)

# Train the word2vec model
w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,   # size of vectors desired
                                   window=5,  # # of words before & after target word to use as context
                                   min_count=2        # min_df
                                   )
w2v_model.build_vocab(X_train)  # prepare the model vocabulary
w2v_model.train(X_train, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
# w2v_model.wv.index_to_key     # all words that Word2Vec model learned a vector for (with min_df=2)
# Find the most similar words to "king" based on word vectors from our trained model
# w2v_model.wv.most_similar('king')
"""[('employee', 0.9108399152755737),
    ('met', 0.9003692269325256),
    ('helping', 0.9003664255142212),
    ('habe', 0.8938229084014893),
    ('irl', 0.8932299613952637),
    ('supplies', 0.8887951374053955),
    ('collect', 0.888152539730072),
    ('citi', 0.8794213533401489),
    ('hence', 0.8784809112548828),
    ('congratulations', 0.8755719065666199)]"""


# Visualizing Word2Vec Embeddings with t-SNE
semantic_similarity(X_train)


# Generate aggregated sentence vectors based on the word vectors for each word in the sentence
words = set(w2v_model.wv.index_to_key )
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test])
    # X_train_vect = array of length 14448, each one an array with shape (1,100)


# Why is the length of the sentence different than the length of the sentence vector?
#for i, v in enumerate(X_train_vect):        # enumerate() keeps count(i) and value(v)
#    print(len(X_train.iloc[i]), len(v))     # length of sentence/# of words  ,  length of vector
        # each word is represented by a size 100 vector. Since there are inconsistencies in the lengths
        # of sentences with their corresponding sentence vectors, we need to make every sentence the
        # same size via element-wise average - take average of all i-th element across the word vectors
        # for each sentence. So a sentence of 5 words that has five size-100 vectors will become
        # averaged to one size-100 vector that represents the whole sentence

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))            # one size-100 vector
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))

# Are our sentence vector lengths consistent? Yes - all size-100!
#for i, v in enumerate(X_train_vect_avg):
#    print(len(X_train.iloc[i]), len(v))



# Instantiate and fit a basic Random Forest model on top of the vectors
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())   # ravel() is faster than flatten() 

# Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test_vect_avg)

from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('For validation data: ')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))



# Let's try it on the 20221012 data
df3 = pd.read_csv('20221012.csv', sep='\t')
df3['Text'] = df3.iloc[:,0] + " " + df3.iloc[:,1]
df3['Text'] = df3['Text'].apply(lambda x: gensim.utils.simple_preprocess(str(x)))

# Encoding the label column
df3['Class'] = df3['Class'].map({'R':1,'r':1,'N':0,'n':0})
X_test3 = df3.Text
y_test3 = df3.Class

# Generate aggregated sentence vectors based on the word vectors for each word in the sentence
X_test_vect3 = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test3])

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_test_vect_avg3 = []
for v in X_test_vect3:
    if v.size:
        X_test_vect_avg3.append(v.mean(axis=0))
    else:
        X_test_vect_avg3.append(np.zeros(100, dtype=float))

# Use the trained model to make predictions on the test data
y_pred3 = rf_model.predict(X_test_vect_avg3)

precision3 = precision_score(y_test3, y_pred3)
recall3 = recall_score(y_test3, y_pred3)
print('For 20221012 test data: ')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred3==y_test3).sum()/len(y_pred3), 3)))


# With a different prediction threshold value
proba3 = rf_model.predict_proba(X_test_vect_avg3)

# threshold set at 0.28
threshold = 0.28
threshprob3 = []
for i in range(len(proba3)):
    if round(proba3[i][1] + (0.5-threshold)) == 0:
        threshprob3.append(0)
    else:
        threshprob3.append(1)        
     
precision33 = precision_score(y_test3, threshprob3)
recall33 = recall_score(y_test3, threshprob3)
print('For 20221012 test data using 0.28 threshold: ')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision33, 3), round(recall33, 3), round((threshprob3==y_test3).sum()/len(threshprob3), 3)))
