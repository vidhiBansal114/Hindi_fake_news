# -*- coding: utf-8 -*-
import nltk as nltk
import pandas as pd
import re
import os
import csv

import tempfile
import tarfile

import numpy as np

from sklearn import preprocessing
from sklearn.metrics import accuracy_score


df=pd.read_csv(r"C:\Users\Vidhi Bansal\Downloads\acm_datasetFinal.csv", encoding="utf8")
df.columns =['heading', 'text', 'link','output']
#df=df.drop(['heading'], axis = 1)

nltk.download('punkt')

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
pattern=r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
punctuation = "!=@#$%^&*()_+<>?:.,;\"-"
for ind in df.index:
        a_list = nltk.tokenize.sent_tokenize(str(df['text'][ind]))
        #print(a_list)
        for i in a_list:
            #print(i)
            if str(df['link'][ind]) in i:
                #print(str(df['link'][ind]),i)
                #print('============')
                text=df['text'][ind]
                text=re.sub(r'http\S+','',text)
                text= re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", text)
                text.replace(r'https?:\/\/\S*', "")
                text= re.sub(r'https?:\/\/\S*', '', text, flags=re.MULTILINE)
                #df['text'][ind]=give_emoji_free_text(df['text'][ind])
                text= re.sub(r'\s*[A-Za-z]+\b', '', text)
                print(text)
                for c in text:
                    if c in punctuation:
                        text = text.replace(c, "")

                text = re.sub(r'[A-Za-z]+', '', text)
                #print(df['text'][ind],'----------')
                querywords = text.split()

                resultwords  = [word for word in querywords if word not in stopword]
                #print(resultwords)
                result=[]
                for k in resultwords:
                    if not k.isalpha():
                        result.append(k)
                text = ' '.join(result)
                print("--",text)

                for c in text:
                    if c in punctuation:
                        text = text.replace(c, "")

                df.iloc[ind, df.columns.get_loc('text')]= text.rstrip()
                #df['text'][ind] = text.rstrip()
                #df['text'][ind] = re.sub(r'[^\w\d\s]+', '', df['text'][ind])

            #print(df['text'][ind])
df['text'].replace('#ERROR!', np.nan, inplace=True)
df= df.dropna(subset=['text'])
#data_new3 = data_new1.copy()                          # Create duplicate of data
df.df(subset = ['text'], inplace = True)
df.to_csv(r"C:\Users\Vidhi Bansal\Downloads\acm_datasetFinal2.csv",index=False)
'''
for i in df['text'].apply(str):
    print('------------')
    a_list = nltk.tokenize.sent_tokenize(i)
    print(a_list)
'''