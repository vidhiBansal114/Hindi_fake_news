# -*- coding: utf-8 -*-
import pandas as pd

import os
import csv

import tempfile
import tarfile

import numpy as np

from sklearn import preprocessing
from sklearn.metrics import accuracy_score



df=pd.read_csv(r"C:\Users\Vidhi Bansal\Downloads\acm_datasetFinal.csv", encoding="utf8")
df.columns =['heading', 'text', 'link','output']
df['text'].replace('', np.nan, inplace=True)

df.dropna(subset=['text'], inplace=True)
df['output'].replace('', np.nan, inplace=True)

df.dropna(subset=['output'], inplace=True)

df['link'].replace('', np.nan, inplace=True)

df.dropna(subset=['link'], inplace=True)
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

df=df.drop(['heading','link'], axis = 1)
print(df.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import numpy as np
#from google.colab import files
#uploaded = files.upload()
file1 = open(r"C:\Users\Vidhi Bansal\Downloads\final_stopwords.txt","r+", encoding="utf8")
Lines = file1.readlines()
stopwords=[]
stopword=[]
for line in Lines:
  stopwords.append(line.splitlines())
for i in stopwords:
  for j in i:
    stopword.append(j)

#file1 = open(r"C:\Users\Vidhi Bansal\Downloads\final_stopwords2.txt","r+", encoding="utf8")
#Lines = file1.readlines()
#stopword3=[]
#for line in Lines:
  #stopword.append(line)
stopword2 = ['','.','!','?', 'अर्थात', 'कुछ', 'तेरी', 'साबुत', 'अपनि', 'हूं', 'काफि', 'यिह', 'जा' ,'दे', 'देकर' ,'रह', 'कह' , 'कर' , 'कहा', 'बात' , 'जिन्हों', 'किर', 'कोई', 'हे', 'कोन', 'रहा', 'सब', 'सो', 'तक', 'इंहें', 'इसकि', 'अपनी', 'दबारा', 'सभि', 'होते', 'भीतर', 'निचे', 'घर', 'उन्हें', 'उन्ह' , 'मेरे' , 'था', 'व', 'इसमें', 'उसी', 'बिलकुल', 'होति', 'गया', 'सकता', 'अपना', 'लिये', 'उसका', 'पर', 'दवारा', 'गए', 'है', 'कितना', 'भि', 'लिए', 'वुह ', 'ना', 'किसि', 'परन्तु', 'किन्हें', 'बहुत', 'भी', 'तुम्हारे', 'निहायत', 'उन्हीं', 'वहिं', 'हैं', 'उन्हों', 'इतयादि','यहाँ', 'तब', 'पूरा', 'क्योंकि', 'कौनसा', 'आप', 'हुअ', 'ऐसे', 'एस', 'कारण', 'अप', 'पहले', 'तुम', 'जेसा', 'तिस', 'लेकिन', 'कहते', 'मगर', 'करता', 'संग', 'सभी', 'जीधर', 'किंहों', 'हि', 'द्वारा', 'हुआ', 'तू', 'जिंहें', 'उसने', 'पास', 'वहां', 'वह', 'किंहें', 'इंहों', 'मुझ', 'कुल', 'तिंहों', 'का', 'मेरी', 'तेरे', 'उनके', 'क्या', 'जहाँ', 'काफ़ी', 'वर्ग', 'वरग','बही', 'ये', 'जिस', 'इसि', 'हुई', 'साम्हने', 'नहिं', 'जैसे', 'वहीं', 'दिया', 'अभी', 'यहि', 'वग़ैरह', 'उनकि', 'न', 'जा','बनि', 'हें', 'यिह ', 'उन', 'को', 'तिन्हों', 'उन्होंने', 'तुझे', 'उसे', 'होने', 'इन्हीं', 'थे', 'उंहिं', 'अपने', 'में', 'फिर','यही', 'नीचे', 'होती', 'तिसे', 'हम', 'यदि', 'सारा', 'कर', 'सकते', 'कोइ', 'और', 'जिंहों', 'तिंहें', 'दूसरे', 'जब', 'रहे','अत', 'मानो', 'जिन', 'बाद', 'उनका', 'किया', 'या', 'उनकी', 'कौन', 'ऐसा', 'सबसे', 'अनुसार', 'दुसरे', 'इन', 'अदि','जिसे', 'उसकी', 'इत्यादि', 'करना', 'यहां', 'हुए', 'तेरा', 'आदि', 'पर  ', 'वाले', 'कहता', 'किन्हों', 'किसे', 'जिन्हें', 'मे','होता', 'करने', 'साभ', 'अभि', 'उसको', 'कई', 'बनी', 'के', 'इन्हें', 'वहाँ', 'कोनसा', 'कइ', 'इनका', 'थि', 'बाला','ऱ्वासा', 'हो', 'उंहें', 'दुसरा', 'वे', 'भितर', 'जेसे', 'एवं', 'अंदर', 'दो', 'साथ', 'करें', 'जिधर', 'तरह', 'उसि', 'इस', 'एसे', 'तिन', 'नहीं', 'से','न','उनको', 'किस', 'किसी', 'इसी', 'मैं', 'यह', 'हुइ', 'ले', 'कि', 'की', 'इसलिये', 'रवासा', 'ने', 'जैसा', 'वह ', 'तिन्हें', 'वुह', 'उस', 'उंहों', 'वगेरह', 'उसके', 'मुझे', 'करते', 'जितना', 'जहां', 'इन्हों', 'इसके', 'होना', 'इसका', 'इंहिं', 'एक', 'जो', 'पे', 'ही', 'तो', 'थी', 'रखें', 'इसे', 'इन ', 'के', 'बहि', 'पुरा', 'ओर', 'इसकी']
stopword=stopword+stopword2
#print(stopword)
import  re
lr = LogisticRegression()
#Makes multiclassifier to binarya
ovr=OneVsRestClassifier(lr)
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(df, test_size=0.2)
#x_train = x_train.reshape(x_train.shape[1:])
#words = [word for word in text.split() if word.lower() not in sw_spacy]
#new_text = " ".join(words)
x_tr=[]
j=[]
for i in x_train['text'].astype(str):
  x = i.split()
  final=[]

  #for w in x if not re.match(r'[A-Z]+', w, re.I)
  #x=list(filter(lambda x: not re.match(r'[a-zA-Z]+', x)

  for j in x:

    #if j not in stopword and not j.isalpha():

      final.append(j)
  s=''
  for x in final:
    s=s+' '+x
  x_tr.append(s)
  #words = [word for word in text.split() if word.lower() not in sw_nltk]
    #x_tr.append(i)
#print(x_tr)
x_ts=[]
for i in x_test['text'].astype(str):
  #if i not in stopword:
    x_ts.append(i)
#x_ts
y_train=x_train['output']

y_test=x_test['output']
#print(x_tr)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
#learn vocabulary and return document-term matrix
x_train=vectorizer.fit_transform(x_tr)
print(x_train)
#transform document to document term matrix
x_test=vectorizer.transform(x_ts)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
#Initializing the classifier Network
x_test = x_test.toarray()
x_train = x_train.toarray()
#y_test = y_test.toarray()
#y_train = y_train.toarray()
#model
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
model = Sequential()
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[EarlyStopping(monitor='accuracy', patience=3, min_delta=0.0001)])
accr = model.evaluate(x_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
'''
x_tr=[]
for i in x_train['text'].astype(str):
  x_tr.append(i)
x_ts=[]
for i in x_test['text'].astype(str):
  x_ts.append(i)
x_ts
y_train=x_train['output']

y_test=x_test['output']
print(x_tr)
'''
#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer=TfidfVectorizer()
#learn vocabulary and return document-term matrix
#x_train=vectorizer.fit_transform(x_tr)
#transform document to document term matrix
#x_test=vectorizer.transform(x_ts)
print(x_train.shape)
print(y_train.shape)
'''
ovr.fit(x_train,y_train)
y_pred=ovr.predict(x_test)
accuracy=[]
from sklearn.metrics import accuracy_score
accuracy.append(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
#gnb

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
ovr=OneVsRestClassifier(clf)
ovr.fit(x_train.todense(),y_train)
y_pred=ovr.predict(x_test.todense())
accuracy.append(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
ovr=OneVsRestClassifier(clf)
ovr.fit(x_train.todense(),y_train)
y_pred=ovr.predict(x_test.todense())
accuracy.append(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=100, random_state=0)
ovr=OneVsRestClassifier(clf)
ovr.fit(x_train.todense(),y_train)
y_pred=ovr.predict(x_test.todense())
accuracy.append(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn import svm
clf = svm.SVC()

ovr=OneVsRestClassifier(clf)
ovr.fit(x_train.todense(),y_train)
y_pred=ovr.predict(x_test.todense())
accuracy.append(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(accuracy)
import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = dict()
algo=["Logistic Regression"," GaussianNB","KNN","Random Forest","SVM"]
for i in range(5):
  data[algo[i]]=accuracy[i]
print(data)
name = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(name, values, color ='maroon',
		width = 0.4)

plt.ylabel("Accuracy in %")
plt.xlabel("ML model")
plt.title("Accuracy with 40% test data")
plt.show()
'''