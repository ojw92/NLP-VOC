# ver. 22021027

# inspired by U Mich bag-of-words, CountVectorizer & tf-idf
    # https://www.coursera.org/lecture/python-text-mining/demonstration-case-study-sentiment-analysis-MJ7g3
    # https://github.com/agniiyer/Applied-Text-Mining-in-Python/blob/master/Case%2BStudy%2B-%2BSentiment%2BAnalysis.ipynb

# Bag-of-words approach with CountVectorizer and tf-idf
# Bag-of-words ignores structure and counts frequency of word
# CountVectorizer lets us use BoW approach by converting text into matrix of token counts (sparse matrix)
# Tf-idf allows us to re-scale features
# tf-idf weighs terms based on their importance in a document
     # Higher weight for higher frequency in a document, but low frequency in the corpus
     # Lower weight for common words, OR rarely used & only occur in long documents

#  % precision,  % recall for  epochs at 0.15 classification threshold
# try for 10 epochs

# =======================================================

# bag-of-words approach - convert text documents into a matrix of token counts


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


seed = 5
np.random.seed(seed)
tf.random.set_seed(seed)    # Each library has its own rng so need to declare it separately for each


# 2MB is file size limit for uploading on Deepnote
df22_1 = pd.read_csv('20221011pre1.csv', sep='\t')     # updated data
df22_2 = pd.read_csv('20221011pre2.csv', sep='\t') 
df22_3 = pd.read_csv('20221011pre3.csv', sep='\t') 
df22_4 = pd.read_csv('20221011pre4.csv', sep='\t') 

# use this to detect unlabeled data
# print(df22_1[df22_1.Class.isna()].index)     # 2736
# print(df22_2[df22_2.Class.isna()].index)
# print(df22_3[df22_3.Class.isna()].index)     # 427
# print(df22_4[df22_4.Class.isna()].index)     # Class column has 'N' and 'n'

df22_1.Class.iloc[2736] = df22_1.Content.iloc[2736]
df22_1.Content.iloc[2736] = ''
df22_3.Class.iloc[427] = df22_3.Content.iloc[427]
df22_3.Content.iloc[427] = ''

    # This was a tricky problem - ome rows of df22.Class showed nan values when doing df22.Class.unique()
    # and df22[df22.Class.isna()], but looking at those rows directly (df22[427] and df[2736]) showed no
    # nan values. Only way to solve the problem was to resolve it at the root, by fixing the dataframes
    # imported via pd.read_csv(). R/N labels were put in 'Content' column, while 'Class' column had nan.

df22 = pd.concat([df22_1, df22_2, df22_3, df22_4], ignore_index=True).drop_duplicates()
    # ignore_index = True to ensure every index number is unique, and there are no dupes
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
# text22 = lower_case(text22)             # lowercase built into CountVectorizer & saves time
# text22 = stopword_remove(text22)        # RNN

# =========================


from sklearn.model_selection import train_test_split


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(text22['Text'], classes22['Class'], random_state=seed)

print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)

semantic_similarity(X_train)


# https://www.coursera.org/lecture/python-text-mining/demonstration-case-study-sentiment-analysis-MJ7g3
# https://github.com/agniiyer/Applied-Text-Mining-in-Python/blob/master/Case%2BStudy%2B-%2BSentiment%2BAnalysis.ipynb


print('========================     CountVectorizer     ========================')
# ========================     CountVectorizer     ========================
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # CountVectorizer converts the collection of every word in the corpus into a master array, where
    # each word is a feature. Each document of the corpus will be an array of numbers that indicate the
    # frequency of words that appear in the document whose locations match those of the master array.
    # Master array = length of total unique words
    # Document array = sparse matrix with same length
    # n-grams can also be done with CountVectorizer, where all consecutive 2-word pairs that appear in the
    # corpus are in the master array.


from sklearn.feature_extraction.text import CountVectorizer



# ====== Test parameters (change things here!)======

min_df = 1
num_feat = []
auc_scores = []
ex_pred = []




print('==========     CountVectorizer, 1-gram     ==========')
# *****  Adding n-grams can explosively increase number of features *****


# Fit the CountVectorizer to the training data
# min_df = 2
vect = CountVectorizer(min_df = min_df,     # min frequency appearance to add word to corpus
        ngram_range = (1,1),  # (min,max) n values for n-grams to be extract & created as features (default=1)
        ).fit(X_train)


print("Length of feature names : ", len(vect.get_feature_names_out()))
print("Every 250th feature :", vect.get_feature_names_out()[::250])     # every 250th feature
# vect.vocabulary_ gives all the unique words and their corresponding index value (column # of sparse matrix)

# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)
     # <4143 x 10501 sparse matrix of type '<class 'numpy.int64'>'
	 # with 163910 stored elements in Compressed Sparse Row format>


# logistic regression works well for high-dimensional sparse data
from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression(max_iter=10000)     # default max_iter=100 and it's too small for our data
        # max_iter = maximum number of iterations taken for the solvers to converge
history = model.fit(X_train_vectorized, y_train,
    # batch_size=64, epochs=2,
    # can also monitor validation loss and metrics at the end of each epoch with the following:
    # validation_data=(x_val, y_val),
        )
        # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                # intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                # penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                # verbose=0, warm_start=False)
# print("training history: \n", history.history)   # only for keras neural network models


# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
# https://stackoverflow.com/questions/44476706/what-is-the-difference-between-keras-model-evaluate-and-model-predict
    # ROC curve (receiver operating characteristic curve) is a graph showing the performance of a
    # classification model by comparing FPR & TPR at all classification thresholds

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

# Predict the transformed test documents
# words that appear in X_test that doesn't in X_train will be ignored
y_score = model.decision_function(vect.transform(X_test))     # distance from hyperplane (SVM)
     # decision_function predicts confidence scores for the points by indicating which which side of
     # the hyperplane (classifier line) the point is at
     # confidence score of a point is proportional to the signed distance of that sample to the hyperplane
     # positive y_score -> 1 as prediction; negative y_score -> 0 as prediction
proba = model.predict_proba(vect.transform(X_test))
pred = model.predict(vect.transform(X_test))     # predictions (0 & 1). switch y_score & pred as needed


print('y_test:', y_test.shape)
print('y_score:', y_score.shape)
print('prediction probability:', proba.shape)
print('AUC score from logistic regression: ', roc_auc_score(y_test, y_score))


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test[:], y_score[:], pos_label=1)    # 'thresholds' go in '_'
roc_auc = auc(fpr, tpr)
print('fpr is', fpr)
print('tpr is', tpr)
print('thresholds', thresholds)
print('also auc is', roc_auc)
    # why is there only 1 threshold value between 0 and 1 for me?
    # cus I'm doing roc_curve(y_test, y_predictions) where y_test & predictions both consist of 0 & 1
    # y_test has 0 & 1; we need to use y_score instead of predictions since we need confidence scores
    # based on ROC ex on scikit-learn, we use decision_function (SVC) to calculate y_score
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    # optimal_idx = thresholds[np.argmax(tpr - fpr)]     # ideal number of thresholds to be used


# ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange",
    lw=lw, label="ROC curve (area = %0.2f)" % roc_auc,)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC(Receiver operating characteristic) Curve")
plt.legend(loc="lower right")
plt.show()

# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names_out())   # feature(unique word) in 0-9,a-z order

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()   # argsort() gives indices of the sorted array
        # .coef_[0] gives coefficients of all features
        # positive = good (correlated to samples classified as 1), negative = bad
        # fyi, sorted(model.coef_[0]) returns array whose coefficients are sorted from smallest to largest

# Find the 10 smallest and 10 largest coefficients
# Remember -ve indices mean the array is read backwards!

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))   # 10 smallest
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))   # 10 largest



# These reviews are treated the same by our current model if order of words is disregarded
print("Predictions for 'not an issue, phone is working' and 'an issue, phone is not working' : \n")
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# We can add context through adding sequences of word features known as n-grams
# bi-gram example = "is working", "not working"
# tri-gram example = "is an issue", "not an issue"



# record values of interest when running multiple tests with diff parameters
num_feat.append(len(vect.get_feature_names_out()))
auc_scores.append(roc_auc_score(y_test, y_score))
ex_pred.append(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))




# ==================================================



print('==========     CountVectorizer, 2-gram     ==========')
# *****  Adding n-grams can explosively increase number of features *****


# Fit the CountVectorizer to the training data
# min_df = 2
vect = CountVectorizer(min_df = min_df,     # min frequency appearance to add word to corpus
        ngram_range = (1,2),  # (min,max) n values for n-grams to be extract & created as features (default=1)
        ).fit(X_train)


print("Length of feature names : ", len(vect.get_feature_names_out()))
print("Every 250th feature :", vect.get_feature_names_out()[::250])     # every 250th feature
# vect.vocabulary_ gives all the unique words and their corresponding index value (column # of sparse matrix)

# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)
     # <4143 x 10501 sparse matrix of type '<class 'numpy.int64'>'
	 # with 163910 stored elements in Compressed Sparse Row format>


# logistic regression works well for high-dimensional sparse data
from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression(max_iter=10000)     # default max_iter=100 and it's too small for our data
        # max_iter = maximum number of iterations taken for the solvers to converge
history = model.fit(X_train_vectorized, y_train,
    # batch_size=64, epochs=2,
    # can also monitor validation loss and metrics at the end of each epoch with the following:
    # validation_data=(x_val, y_val),
        )
        # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                # intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                # penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                # verbose=0, warm_start=False)
# print("training history: \n", history.history)   # only for keras neural network models


# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
# https://stackoverflow.com/questions/44476706/what-is-the-difference-between-keras-model-evaluate-and-model-predict
    # ROC curve (receiver operating characteristic curve) is a graph showing the performance of a
    # classification model by comparing FPR & TPR at all classification thresholds

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

# Predict the transformed test documents
# words that appear in X_test that doesn't in X_train will be ignored
y_score = model.decision_function(vect.transform(X_test))     # distance from hyperplane (SVM)
     # decision_function predicts confidence scores for the points by indicating which which side of
     # the hyperplane (classifier line) the point is at
     # confidence score of a point is proportional to the signed distance of that sample to the hyperplane
     # positive y_score -> 1 as prediction; negative y_score -> 0 as prediction
proba = model.predict_proba(vect.transform(X_test))
pred = model.predict(vect.transform(X_test))     # predictions (0 & 1). switch y_score & pred as needed


print('y_test:', y_test.shape)
print('y_score:', y_score.shape)
print('prediction probability:', proba.shape)
print('AUC score from logistic regression: ', roc_auc_score(y_test, y_score))


# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test[:], y_score[:], pos_label=1)    # 'thresholds' go in '_'
roc_auc = auc(fpr, tpr)
print('fpr is', fpr)
print('tpr is', tpr)
print('thresholds', thresholds)
print('also auc is', roc_auc)
    # why is there only 1 threshold value between 0 and 1 for me?
    # cus I'm doing roc_curve(y_test, y_predictions) where y_test & predictions both consist of 0 & 1
    # y_test has 0 & 1; we need to use y_score instead of predictions since we need confidence scores
    # based on ROC ex on scikit-learn, we use decision_function (SVC) to calculate y_score
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    # optimal_idx = thresholds[np.argmax(tpr - fpr)]     # ideal number of thresholds to be used


# ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange",
    lw=lw, label="ROC curve (area = %0.2f)" % roc_auc,)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC(Receiver operating characteristic) Curve")
plt.legend(loc="lower right")
plt.show()

# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names_out())   # feature(unique word) in 0-9,a-z order

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()   # argsort() gives indices of the sorted array
        # .coef_[0] gives coefficients of all features
        # positive = good (correlated to samples classified as 1), negative = bad
        # fyi, sorted(model.coef_[0]) returns array whose coefficients are sorted from smallest to largest

# Find the 10 smallest and 10 largest coefficients
# Remember -ve indices mean the array is read backwards!

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))   # 10 smallest
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))   # 10 largest



# These reviews are treated the same by our current model if order of words is disregarded
print("Predictions for 'not an issue, phone is working' and 'an issue, phone is not working' : \n")
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# We can add context through adding sequences of word features known as n-grams
# bi-gram example = "is working", "not working"
# tri-gram example = "is an issue", "not an issue"



# record values of interest when running multiple tests with diff parameters
num_feat.append(len(vect.get_feature_names_out()))
auc_scores.append(roc_auc_score(y_test, y_score))
ex_pred.append(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# ==================================================


print('Number of features for each n-gram:', num_feat)
print('AUC scores for each n-gram:', auc_scores)
print('Example prediction outcome for each n-gram:', ex_pred)




# test out accuracy on VOC data here!

df3 = pd.read_csv('20221012.csv', sep='\t')    # import 20221013 issues
text3, classes3 = data_split(df3)
# text3 = stopword_remove(text3)        # RNN
    # most preprocessing steps redundant - vect.transform() will do the job

textlist3 = text3.values.tolist()     # each VOC text is a list of 1 string. Need to flatten list of lists
flat3 = [item for sublist in textlist3 for item in sublist]

print("Actual label for 20221012 data: ")
print(classes3)
print("Y_scores for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
print(model.decision_function(vect.transform(flat3)))
print("Prediction probabilities for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
print(model.predict_proba(vect.transform(flat3)))
print("Log prediction probabilities for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
print(model.predict_log_proba(vect.transform(flat3)))
print("Predictions for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
print(model.predict(vect.transform(flat3)))



print('Showing semantic similarities for whole dataset, not just training set')
semantic_similarity(text22)

y_pred3 = model.predict(vect.transform(flat3))
from sklearn.metrics import precision_score, recall_score
precision3 = precision_score(classes3, y_pred3)
recall3 = recall_score(classes3, y_pred3)
print('For validation data: ')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision3, 3), round(recall3, 3), round((y_pred3==classes3).sum()/len(y_pred3), 3)))



end = timer()
print("%4f seconds, %4f minutes elapsed" % (float(end-start), float((end-start)/60)))

