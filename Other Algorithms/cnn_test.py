
# ver. 20221012

# ========================= Test today's data =========================
# Run this right after running the above code


start = timer()



# Try feeding today's data to see how much of it the algorithm gets right!


df3 = pd.read_csv('20220407pre.csv', sep='\t')    # import 20220407 issues
#df3 = df3.sample(frac=1, random_state=seed)     # shuffle order of entries (all of R's near the top)


text3, classes3 = data_split(df3)
text3 = hep_remove(text3)
text3 = emoji_remove(text3)
text3 = punc_remove(text3)
text3 = lower_case(text3)
text3 = stopword_remove(text3, model="CNN")        # CNN
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
thresh = 0.15
pred = []
y_pred = model.predict(padded_x3)
y_pred1 = (model.predict(padded_x3)>thresh).astype('int32')

for i in range(len(padded_x3)):
    print("VOC = blah, Predicted = %s --> %s" % (y_pred[i], y_pred1[i]))  # padded_x3[i], goes in the blah
    pred.append(int(y_pred1[i]))

print('Percent predicted R with threshold %.2f : %d / %d = %.4f' % 
        (thresh, sum(pred), len(pred), (sum(pred)/len(pred))))


plt.plot(range(len(padded_x3)), y_pred)
plt.xlabel('voc')
plt.ylabel('predictions')


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
