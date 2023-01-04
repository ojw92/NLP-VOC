# Model for S22 VOC only, but with added preprocessing steps
# ver. 20221012


# ==========================================================


# CNN model train

# try min_df & n-grams!

# ==========================================================



from timeit import default_timer as timer

start = timer()


import numpy as np
import tensorflow as tf
from numpy import array

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout, Activation
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical


seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


import pandas as pd

# 2MB is file size limit for uploading on Deepnote
df22_1 = pd.read_csv('20221011pre1.csv', sep='\t')     # updated data
df22_2 = pd.read_csv('20221011pre2.csv', sep='\t') 
df22_3 = pd.read_csv('20221011pre3.csv', sep='\t') 
df22_4 = pd.read_csv('20221011pre4.csv', sep='\t') 
df22 = pd.concat([df22_1, df22_2, df22_3, df22_4])
    # df22 = pd.read_csv('s22train20220405.csv', sep='\t')    # old R & N-labeled issues
df22 = df22.sample(frac=1, random_state=seed)     # shuffle the order so R's are not all at the top



# https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91
#      Remove stopwords, Apply lemmatization, and other preprocessing ideas I have
# Define my very own "Super Function" that filters and preprocesses words from VOC
# 1 - remove hyperlinks, emails & pricetags (done!)
# 2 - remove punctuations
# 3 - replace country/city names with "place", and other proper nouns with generic nouns
# 4 - apply lower case (done!)
# 5 - remove stopwords
# 6 - lemmatize (and stemming?)

# ==================================================

text22, classes22 = data_split(df22)
text22 = hep_remove(text22)
text22 = emoji_remove(text22)
text22 = punc_remove(text22)
text22 = lower_case(text22)
text22 = stopword_remove(text22, model="CNN")        # CNN
text22 = lemmatize_df(text22)

# ==================================================


from sklearn.feature_extraction.text import TfidfVectorizer

# term frequency-inverse document frequency

# Features with high tfidf are usually used in specific types of documents but rarely used across all documents.
# Features with low tfidf are generally used across all documents in the corpus.

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
# Each token needs to appear in at least 5 documents to become a part of the vocabulary.
vect = TfidfVectorizer(min_df=2).fit(text22)
print(len(vect.get_feature_names_out()))
print(vect)




# ==================================================

# Final check
print(type(text22))           # <class 'numpy.ndarray'>
print(type(text22.iloc[0]))        # <class 'numpy.str_'>
print(type(classes22))        # <class 'pandas.core.series.Series'>
print(type(classes22.iloc[0]))     # <class 'pandas.core.series.Series'>
print(classes22.iloc[0].dtype)     # int64




padded_x22, word_size22, max_x22 = tokenize_pad_words(text22)



#      k-fold cross validation
# =========================

# divide data into sections to be used for training & testing and repeat with diff combinations
from sklearn.model_selection import StratifiedKFold
n_fold = 2     # k-1 for train data (9), 1 for test data (1)
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
accuracy = []

# =========================


#      Save well-trained models
# =========================

import os
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
modelpath = "./model/{epoch:02d}-{loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

# =========================


#      Model fitting, evaluating & graphing
# =========================

# Incorporate word embedding to compress VOC vectors with too many words
modelaccu = 0
modeltouse = Sequential()
epochs = 6
"""
# for k-fold

# train & test are randomly chosen indices (arrays of int)
for train, test in skf.split(padded_x22, classes22):
    model = Sequential()
    model.add(Embedding(word_size22, 20, input_length=max_x22))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))     # CNN
    model.add(MaxPooling1D(pool_size=4))
    # model.add(LSTM(100))       # RNN          Try it with just CNN, then add RNN later
    # model.add(Flatten())
    # model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(padded_x22[train], classes22[train], epochs=epochs,        batch_size=125, 
                        validation_split=0.2,           # (8:1):1  train-valid-test
                        callbacks=[early_stopping_callback, checkpointer])
    testeval22 = model.evaluate(padded_x22[test], classes22[test])      # [0]=loss, [1]=accuracy
    print('\nAccuracy : %.4f' % testeval22[1])
    accuracy.append(float('%.4f' % testeval22[1]))

    if testeval22[1] > modelaccu:
        modelaccu = testeval22[1]
        modeltouse = model

    # add plots of error & accuracy
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Error & Accuracy')
    y_vloss = history.history['val_loss']     # validation error
    y_loss = history.history['loss']          # training error
    x_len = np.arange(len(y_loss))
    ax1.plot(x_len, y_vloss, marker='.', c='red', label='Valid_loss')
    ax1.plot(x_len, y_loss, marker='.', c='blue', label='Train_loss')
    ax1.plot(epochs, float('%.4f' % testeval22[0]), marker='o', c='green', label='Test_loss')
    ax1.legend(loc='upper right')
    ax1.grid()
    #plt.xlabel('epoch')
    #plt.ylabel('loss')

    y_vaccu = history.history['val_accuracy']     # validation error
    y_accu = history.history['accuracy']          # training error
    ax2.plot(x_len, y_vaccu, marker='.', c='red', label='Valid_accuracy')
    ax2.plot(x_len, y_accu, marker='.', c='blue', label='Train_accuracy')
    ax2.plot(epochs, float('%.4f' % testeval22[1]), marker='o', c='green', label='Test_accuracy')
    ax2.legend(loc='upper right')
    ax2.grid()
    plt.show()
    
model = modeltouse         # our model is the one that had the best accuracy


print('\n %.f fold accuracy : ' % n_fold, accuracy)
print('average accuracy : %.4f' % (sum(accuracy)/len(accuracy)))
"""

# just train without k-fold
model = Sequential()
model.add(Embedding(word_size22, 20, input_length=max_x22))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))     # CNN
model.add(MaxPooling1D(pool_size=4))
# model.add(LSTM(100))       # RNN          Try it with just CNN, then add RNN later
model.add(Flatten())        # need this if using CNN, but not LSTM
# model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(padded_x22, classes22, epochs=epochs,        batch_size=125, 
                    validation_split=0.2,           # (8:1):1  train-valid-test
                    callbacks=[early_stopping_callback, checkpointer])
testeval22 = model.evaluate(padded_x22, classes22)      # [0]=loss, [1]=accuracy
print('\nAccuracy : %.4f' % testeval22[1])
accuracy.append(float('%.4f' % testeval22[1]))

# add plots of error & accuracy
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Error & Accuracy')
y_vloss = history.history['val_loss']     # validation error
y_loss = history.history['loss']          # training error
x_len = np.arange(len(y_loss))
ax1.plot(x_len, y_vloss, marker='.', c='red', label='Valid_loss')
ax1.plot(x_len, y_loss, marker='.', c='blue', label='Train_loss')
ax1.plot(epochs, float('%.4f' % testeval22[0]), marker='o', c='green', label='Test_loss')
ax1.legend(loc='upper right')
ax1.grid()
#plt.xlabel('epoch')
#plt.ylabel('loss')

y_vaccu = history.history['val_accuracy']     # validation error
y_accu = history.history['accuracy']          # training error
ax2.plot(x_len, y_vaccu, marker='.', c='red', label='Valid_accuracy')
ax2.plot(x_len, y_accu, marker='.', c='blue', label='Train_accuracy')
ax2.plot(epochs, float('%.4f' % testeval22[1]), marker='o', c='green', label='Test_accuracy')
ax2.legend(loc='upper right')
ax2.grid()
plt.show()





# We get a feel for the performance if we were to train on the whole dataset would,
# but we need to now make a model using the whole dataset - do that later
# Let's we evaluate on the data.

end = timer()
#elapsed = end-start
print((end - start), "seconds have passed")
# print("%4f minutes have passed" % elapsed/60)