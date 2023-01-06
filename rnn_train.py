# Model for S22 VOC only, but with added preprocessing steps
# ver. 20221025

# =======================================================

# RNN model train
#  minor word removal
# 37.5 % precision, 28.125 % recall for 6 epochs at 0.15 classification threshold - 2022.4.15.
# try for lower threshold
# try for 4 epochs

# =======================================================



from timeit import default_timer as timer

start = timer()

import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import array
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem import WordNetLemmatizer     # Lemmatize – rocks, better  ->  rock, good

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout, Activation
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

seed = 5
np.random.seed(seed)
tf.random.set_seed(seed)    # Each library has its own rng so need to declare it separately for each


# 2MB is file size limit for uploading on Deepnote
df22_1 = pd.read_csv('20221011pre1.csv', sep='\t')     # updated data
df22_2 = pd.read_csv('20221011pre2.csv', sep='\t') 
df22_3 = pd.read_csv('20221011pre3.csv', sep='\t') 
df22_4 = pd.read_csv('20221011pre4.csv', sep='\t') 

# use this to detect unlabeled data
#print(df22_1[df22_1.Class.isna()].index)     # 2736
#print(df22_2[df22_2.Class.isna()].index)
#print(df22_3[df22_3.Class.isna()].index)     # 427
#print(df22_4[df22_4.Class.isna()].index)     # Class column has 'N' and 'n'

df22_1.Class.iloc[2736] = df22_1.Content.iloc[2736]
df22_1.Content.iloc[2736] = ''
df22_3.Class.iloc[427] = df22_3.Content.iloc[427]
df22_3.Content.iloc[427] = ''

    # This was a tricky problem - ome rows of df22.Class showed nan values when doing df22.Class.unique()
    # and df22[df22.Class.isna()], but looking at those rows directly (df22[427] and df[2736]) showed no
    # nan values. Only way to solve the problem was to resolve it at the root, by fixing the dataframes
    # imported via pd.read_csv(). R/N labels were put in 'Content' column, while 'Class' column had nan.

df22 = pd.concat([df22_1, df22_2, df22_3, df22_4], ignore_index=True).drop_duplicates()
    # ignore_index = True to ensure every index number is unique, and there are no duplicate index numbers
    # df22 = pd.read_csv('s22train20220405.csv', sep='\t')    # old R & N-labeled issues
df22 = df22.sample(frac=1, random_state=seed)     # shuffle the order so R's are not all at the top
    # .sample returns a sample of rows (frac=1 is for all rows)


# https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91
#      Remove stopwords, Apply lemmatization, and other preprocessing ideas I have
# Define my very own "Super Function" that filters and preprocesses words from VOC
# 1 - remove hyperlinks, emails & pricetags (done!)
# 2 - remove punctuations
# 3 - replace country/city names with "place", and other proper nouns with generic nouns
# 4 - apply lower case (done!)
# 5 - remove stopwords
# 6 - lemmatize (and stemming?)

# =========================

text22, classes22 = data_split(df22)
text22, classes22 = drop_dupe_text(text22, classes22)
text22 = hep_remove(text22)
text22 = emoji_remove(text22)
text22 = punc_remove(text22)
# text22 = lower_case(text22)           # Tokenizer() has lowercase feature - compare later!
text22 = stopword_remove(text22)        # RNN


# ==================================================


text22 = lemmatize_df(text22)
# text22 = stemming_df(text22)


# ==================================================
"""
# Final check
print(type(text22))           # <class 'numpy.ndarray'>
print(type(text22.iloc[0]))        # <class 'numpy.str_'>
print(type(classes22))        # <class 'pandas.core.series.Series'>
print(type(classes22.iloc[0]))     # <class 'pandas.core.series.Series'>
print(classes22.iloc[0].dtype)     # int64
"""


padded_x22, word_size22, max_x22 = tokenize_pad_words(text22)


# Split data into training and test sets ()
Y = to_categorical(classes22)
X_train, X_val, y_train, y_val = train_test_split(padded_x22, Y, test_size=0.2, random_state=seed)

print('padded X_train first entry:\n\n', X_train[0])
print('\n\n padded X_train shape: ', X_train.shape)



# =========================

"""
#      k-fold cross validation (too slow to implement in RNN)
# =========================

# divide data into sections to be used for training & testing and repeat with diff combinations
from sklearn.model_selection import StratifiedKFold
n_fold = 2     # k-1 for train data (9), 1 for test data (1)
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
"""
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
#modelaccu = 0
#modeltouse = Sequential()
epochs = 3

print("Training the model at 8-2 train-val split: ")
# just train without k-fold
model = Sequential()
model.add(Embedding(word_size22, 20, input_length=max_x22))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))     # CNN
#model.add(MaxPooling1D(pool_size=4))
#model.add(Flatten())        # need this if using CNN, but not LSTM
#model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(padded_x22, classes22, epochs=epochs,        batch_size=125, 
                    validation_split=0.2,           # (8:1):1  train-valid-test
                    callbacks=[early_stopping_callback, checkpointer])
testeval22 = model.evaluate(padded_x22, classes22)      # [0]=loss, [1]=accuracy
print('\nAccuracy : %.4f' % testeval22[1])
accuracy.append(float('%.4f' % testeval22[1]))     # 'accuracy' initially defined under k-fold part of code


y_pred = model.predict()

from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
print('For validation data: ')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_val).sum()/len(y_pred), 3)))


# add plots of error & accuracy
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
# plt.xlabel('epoch')
# plt.ylabel('loss')

y_vaccu = history.history['val_accuracy']     # validation error
y_accu = history.history['accuracy']          # training error
ax2.plot(x_len, y_vaccu, marker='.', c='red', label='Valid_accuracy')
ax2.plot(x_len, y_accu, marker='.', c='blue', label='Train_accuracy')
ax2.plot(epochs, float('%.4f' % testeval22[1]), marker='o', c='green', label='Test_accuracy')
ax2.legend(loc='upper right')
ax2.grid()
plt.show()


print('Showing semantic similarities for whole dataset, not just training set')
semantic_similarity(text22)


# We get a feel for the performance if we were to train on the whole dataset would,
# but we need to now make a model using the whole dataset - do that later
# Let's we evaluate on the data.

end = timer()
print("%4f minutes have passed" % float((end-start)/60))

