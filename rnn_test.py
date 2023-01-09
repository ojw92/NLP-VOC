
# ========================= Test today's data =========================
# ver.20221026


start = timer()



# Try feeding today's data to see how much of it the algorithm gets right!


df3 = pd.read_csv('20221012.csv', sep='\t')    # import 20221013 issues
#df3 = df3.sample(frac=1, random_state=seed)     # shuffle order of entries (all of R's near the top)

text3, classes3 = data_split(df3)
text3 = hep_remove(text3)
text3 = emoji_remove(text3)
text3 = punc_remove(text3)
# text3 = lower_case(text3)
text3 = stopword_remove(text3)        # RNN
text3 = lemmatize_df(text3)


# ==================================================

# Final check
print(type(text3))                # <class 'pandas.core.frame.DataFrame'>
print(type(text3.iloc[0]))        # <class 'pandas.core.series.Series'>
print(type(classes3))             # <class 'pandas.core.series.Series'>
print(type(classes3.iloc[0]))     # <class 'numpy.int64'>
print(classes3.iloc[0].dtype)     # int64


# ==================================================


padded_x3, word_size3, max_x3 = tokenize_pad_words(text3, max_xref=max_x22)



# Use the model to test the data
from tensorflow.keras.models import load_model
# model = load_model(./model/blah-blah.hdf5)
testeval3 = model.evaluate(padded_x3, classes3)     


# Accuracy here means nothing; classes weren't labeled yet. It's just a placeholder for our functions

print('\nAccuracy : %.4f' % testeval3[1])

# add plots of error & accuracy
import matplotlib.pyplot as plt
y_vloss = history.history['val_loss']     # validation error
y_loss = history.history['loss']          # training error
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Valid_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Train_loss')
plt.plot(epochs, float('%.4f' % testeval3[0]), marker='o', c='green', label='Test_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



# ========== This is the main output we care about - the predictions ==========



# Make a prediction & show the inputs and predicted outputs
thresh = 0.13
y_prob3 = model.predict(padded_x3)
y_pred3 = (model.predict(padded_x3)>thresh).astype('int32')

for i in range(len(padded_x3)):
    print("VOC = %s, Predicted = %s --> %s, Actual = %s" 
            % (df3.Title.iloc[i][:30], y_prob3[i], y_pred3[i], classes3.Class.iloc[i]))  # padded_x3[i], goes in the blah
    pred.append(int(y_pred3[i]))

y_prob3 = [item for sublist in y_prob3 for item in sublist]     # flatten the array of arrays
y_pred3 = [item for sublist in y_pred3 for item in sublist]   # flatten the array of arrays
y_probpred = pd.DataFrame({"y_prob3" : y_prob3, "y_pred3" : y_pred3})


print('Percent predicted R with threshold %.2f : %d / %d = %.4f' % 
        (thresh, sum(y_pred3), len(y_pred3), (sum(y_pred3)/len(y_pred3))))


plt.scatter(range(len(padded_x3)), y_prob3)
plt.xlabel('voc')
plt.ylabel('predictions')


from sklearn.metrics import precision_score, recall_score
precision3 = precision_score(classes3, y_pred3)
recall3 = recall_score(classes3, y_pred3)
print('For test data: ')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision3, 3), round(recall3, 3), round((y_probpred.y_pred3==classes3.Class).sum()/len(y_pred3), 3)))


end = timer()
print("%4f seconds have passed" % float(end-start))



#print('\nAccuracy : %.4f' % testeval22[1])
#accuracy.append(float('%.4f' % testeval22[1]))



# I may need my own word bank:
    # get rid of price tags, pronouns, etc
    # different variation of model names into 1 word ('s22') - distinguish Ultra, Plus if needed (prob not)
# use of country, city names to determine snapdragon vs exynos
    # https://www.nltk.org/howto/chat80.html  (from https://www.nltk.org/book/ch02.html)
# Useful techniques to apply + test case:     https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# also filter out proper nouns using NER(Named Entity Recognition)