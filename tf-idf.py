
# Using tf-idf with logistic regression for text classification
# Train on a portion of given 18046 VOC data (Title & Content of each community post) to make
# predictions on which post reports a technical issue of mobile device


import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import array
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score

from timeit import default_timer as timer
start = timer()

seed = 5
np.random.seed(seed)
tf.random.set_seed(seed)    # Each library has its own rng so need to declare it separately for each


# Upload dataset
df22 = pd.read_csv('S22_train20221011.csv', sep='\t', index_col=0).drop_duplicates()

# Fix rows where values are shifted by 1 column
shift_rows = df22.loc[df22[df22.Class.apply(lambda x: x == 'None')].index]   # rows where columns shifted to left
shift_values = shift_rows[df22.columns[1:-1]]
shift_rows[df22.columns[2:]] = shift_rows[df22.columns[1:-1]]
shift_rows.Content = ''
df22.loc[df22[df22.Class.apply(lambda x: x == 'None')].index] = shift_rows

# Filter training data to desired dates
df22 = df22[df22.Date.apply(lambda x: datetime.strptime(x,'%m/%d/%Y %H:%M').date() > datetime(2022,2,21).date())]

# For class imbalance, use roughly same ratio of R & N
df22 = pd.concat([df22[df22.Class=='R'], df22[df22.Class=='N'].iloc[::4, :]])
df22 = df22.sample(frac=1, random_state=seed)     # shuffle the order so R's are not all at the top

# Use the predefined preprocessing steps to clean up the text
text22, classes22 = data_split(df22)
text22, classes22 = drop_dupe_text(text22, classes22)
text22 = hep_remove(text22)
text22 = emoji_remove(text22)
text22 = punc_remove(text22)
text3 = stopword_remove(text3)        

# Split data into training and test sets. Stratify since there are 4 times as many 0's as 1's in labels
X_train, X_test, y_train, y_test = train_test_split(text22['Text'], classes22['Class'], 
                                    test_size=0.1, stratify=classes22['Class'], random_state=seed)

print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)

# ================ Test parameters ================

min_df = 15
num_feat = []
auc_scores = []
ex_pred = []

# =================================================

# Tf-idf vectorization with bi-grams
print('==========     Tf-idf, 2-gram     ==========')
vect = TfidfVectorizer(min_df=min_df,
        ngram_range = (1,2),  # (min,max) n values for n-grams to be extract & created as features (default=1)
        ).fit(X_train)     # vect.fit(x).transform(x) = vect.fit_transform(x)
print("how many features are there? ", len(vect.get_feature_names_out()))
print("type of vect: ", type(vect))
print("vect: ", vect)

X_train_vectorized = vect.transform(X_train)
print("type of X_train_vectorized: ", type(X_train_vectorized))
print("shape of X_train_vectorized:", X_train_vectorized.shape)
print("X_train_vectorized: ", X_train_vectorized)

# Initialize logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_vectorized, y_train)

# Predict the transformed test documents
# words that appear in X_test that doesn't in X_train will be ignored
y_score = model.decision_function(vect.transform(X_test))     # distance from hyperplane (SVM)
proba = model.predict_proba(vect.transform(X_test))
pred = model.predict(vect.transform(X_test))     # predictions (0 & 1)

print('y_test:', y_test.shape)
print('y_score:', y_score.shape)
print('prediction probability:', proba.shape)
print('AUC score from logistic regression: ', roc_auc_score(y_test, y_score))

# Create ROC curve and compute AUC
fpr, tpr, thresholds = roc_curve(y_test[:], y_score[:], pos_label=1)    # 'thresholds' go in '_'
roc_auc = auc(fpr, tpr)
print('fpr is', fpr)
print('tpr is', tpr)
print('thresholds', thresholds)
print('also auc is', roc_auc)

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

feature_names = np.array(vect.get_feature_names_out())

# Print words with smallest & highest tf-idf scores. Higher score = more important, relevant word
sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
print('Smallest tf-idf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tf-idf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
# Print words with smallest & highest coefficients. Larger coef = contributes more to classifying as 1
sorted_coef_index = model.coef_[0].argsort()
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# Predict 
print("Predictions for 'not an issue, phone is working' and 'an issue, phone is not working' : \n")
print(model.predict_proba(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))

# record values of interest when running multiple tests with diff parameters
num_feat.append(len(vect.get_feature_names_out()))
auc_scores.append(roc_auc_score(y_test, y_score))
ex_pred.append(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))

print('Number of features for each n-gram:', num_feat)
print('AUC scores for each n-gram:', auc_scores)
print('Example prediction outcome for each n-gram:', ex_pred)

# =========================================================================================================
"""
"""
# test out accuracy on VOC data here!





df3 = pd.read_csv('20221017.csv', sep='\t')    # import 20221013 issues
text3, classes3 = data_split(df3)
text3 = hep_remove(text3)
text3 = emoji_remove(text3)
text3 = punc_remove(text3)
text3 = stopword_remove(text3)       
    # most preprocessing steps redundant - vect.transform() will do the job

textlist3 = text3.values.tolist()     # each VOC text is a list of 1 string. Need to flatten list of lists
flat3 = [item for sublist in textlist3 for item in sublist]

# thresh=0.50
y_proba3 = model.predict_proba(vect.transform(flat3))
y_prob3 = []
for i in range(len(y_proba3)):
    y_prob3.append(y_proba3[i][1])
# y_pred3 = (model.predict_proba(vect.transform(flat3))>thresh).astype('int32')  # use for other threshold values
y_pred3 = model.predict(vect.transform(flat3))

# Uncomment below to see y-scores, prediction probabilities, and predictions
#print("Actual label for 20221012 data: ")
#print(classes3)
#print("Y_scores for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
#print(model.decision_function(vect.transform(flat3)))
#print("Prediction probabilities for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
#print(model.predict_proba(vect.transform(flat3)))
#print("Log prediction probabilities for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
#print(model.predict_log_proba(vect.transform(flat3)))
#print("Predictions for 20221012 data using 2-gram, 15 min_df Tf-idf: ")
#print(model.predict(vect.transform(flat3)))

# Print first 30 characters of each VOC, algorithm's prediction probability and actual value
for i in range(len(flat3)):
    print("VOC = %s, Predicted = %s, Actual = %s" 
            % (df3.Title[i][:30], y_prob3[i], classes3.Class[i]))
    # pred.append(int(y_pred3[i]))

print('Percent predicted R with threshold %.2f : %d / %d = %.4f' % 
        (thresh, sum(y_pred3), len(y_pred3), (sum(y_pred3)/len(y_pred3))))

# Create scatterplot of VOC predictions with classifier line
x = np.array(range(len(y_prob3)))
y = np.array(y_prob3)
number_of_R = sum(classes3.Class==1)
scatterplot_VOC(x=x, y=y, number_of_R=number_of_R, threshold=thresh)


# y_probactual = pd.DataFrame({"pred" : y_prob3, "actual" : classes3.Class})
precision3 = precision_score(classes3, y_pred3)
recall3 = recall_score(classes3, y_pred3)
print('For test data: ')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision3, 3), round(recall3, 3), round((y_pred3==classes3.Class).sum()/len(y_pred3), 3)))



end = timer()
print("%4f minutes have passed" % float((end-start)/60))
