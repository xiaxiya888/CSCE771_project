
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
df=pd.read_csv('yelp_academic_dataset_review.csv',nrows=100000)
# df.head()
# print(df.shape)
df_filtered=df[df['stars'] !=3] 
# print(df_filtered.shape)

print(df_filtered.describe().T)


#df['text'][0]
#print(df['text'][0])

#text=list(df['text'])
stars=list(df_filtered['stars'])


#sns.countplot(x = df['stars'])
#plt.show()

label=[]

for item in stars:
    if item>= 4:
        y=1
    else:
        y=0
    label.append(y)
# print(type(label))
# all_count=len(label)
# count=0
# for item in label:
# 	if item==1:
# 		count+=1
        
# print(count)
# print(all_count)
# #count=64471
# sns.countplot(x = label)
# plt.show()


#bag of words
vectorizer=CountVectorizer(ngram_range=(2,2))


# #TF-IDF
transformer=TfidfTransformer()
x=transformer.fit_transform(vectorizer.fit_transform(df_filtered['text'].values.astype('U')))
x_train, x_test, train_y, test_y = model_selection.train_test_split(x,label, test_size=0.2, random_state=42)

print("\t\t\tFeatures Shapes:")
print("Train set: \t\t{}".format(x_train.shape),
      "\nTest set: \t\t{}".format(x_test.shape))



#fit a NB model


nb = MultinomialNB()
model=nb.fit(x_train, train_y)




def model_test(X_test, Y_test, svr):
    predictY = svr.predict(X_test)
    result = sum(predictY == Y_test)
    return float(result) / len(Y_test)


#print(classification_report(test_y, model.predict(x_test)))

accuracy_test2 = model_test(x_test, test_y, model)
print(accuracy_test2)

# def binary_classification_performance(test_y, y_pred):

#     accuracy = round(accuracy_score(y_pred = y_pred, y_true = test_y),2)
#     precision = round(precision_score(y_pred = y_pred, y_true = test_y),2)
#     recall = round(recall_score(y_pred = y_pred, y_true = test_y),2)
 
    


#     result = pd.DataFrame({'Accuracy' : [accuracy],
#                          'Precision (or PPV)' : [precision],
#                          'Recall (senitivity or TPR)' : [recall],
#                         })
#     return result


# print(binary_classification_performance(test_y, model.predict(x_test)))
