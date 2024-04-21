# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as colors
from sklearn.cluster import KMeans
#df = pd.read_csv(r"D:/dissertation/S1File.csv")
#print(df)
# Importing Modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Loading dataset
#df = pd.read_csv(r"D:\dissertation\S1File.csv")
#print(df['IFBSCP'])
#df.loc[df['IFBSCP']=='#DIV/0!', ['IFBSCP']] = 0
col=['No. of cit. past 2 years','No. of self-cit. past 2 years','No. of cit. preceding 5 years','No. of self-cit. preceding 5 years','IFBSCP']
#print(df['IFBSCP'])
#df=df.loc[:,col]
#kmeans = KMeans(2)
#kmeans.fit(df)
#identified_clusters = kmeans.fit_predict(df)

# create a figure and axis
#fig, ax = plt.subplots()

# scatter the sepal_length against the sepal_width
#ax.scatter(df['No. of cit. past 2 years'], df['No. of self-cit. past 2 years'])
# set a title and labels
#ax.set_title('Citation dataset')
#ax.set_xlabel('No. of cit. past 2 years')
#ax.set_ylabel('No. of self-cit. past 2 years')
#plt.show()
#plt.scatter(df['No. of cit. past 2 years'], df['No. of cit. preceding 5 years'])
#plt.show()
file1 = open('D:\dissertation\dataset\DBLP-citation-network-Oct-19 (1)\DBLPOnlyCitationOct19.txt', "r",encoding='utf-8')
count = 0

# Using for loop
print("Using for loop")
c=0
data=[]
curr=[]
for line in file1:
        curr.append(line.rstrip())
        if line=='\n':
                data.append(curr)
                curr=[]
c=0
final=[]
df = pd.DataFrame(columns = ['title', 'authors', 'year','PublicationVenue','index','references','abstract'])
for i in data:
        title = ""
        authors = ""
        year = ""
        PublicationVenue = ""
        index = ""
        references = []
        abstract = ""
        curr=[]
        for j in i:
                #if j=='\n':
                        #continue

                if j[:2]=="#*":
                        title=j[2:]
                        #print(title)
                if j[:2]=="#@":
                        authors=j[2:]
                if j[:2]=="#t":
                        year=j[2:]
                if j[:2]=="#c":
                        PublicationVenue=j[2:]
                if j[:6]=="#index":
                        index=j[6:]
                if j[:2]=="#%":
                        references.append(j[2:])
                if j[:2]=="#!":
                       abstract=j[2:]
        df2={'title':title, 'authors':authors, 'year':year,'PublicationVenue':PublicationVenue,'index':index,
                   'references':references,'abstract':abstract}
        df=pd.DataFrame(df2)
        #print(df2)
        curr.append(title)
        curr.append(authors)
        curr.append(year)
        curr.append(PublicationVenue)
        curr.append(index)
        curr.append(references)
        curr.append(abstract)
        #print(curr)
        final.append(curr)


        #with open("D:\dissertation\dataset\citation-network1\outputacm.csv", "a", newline="") as f:
                #writer = csv.writer(f)
                #writer.writerows(curr)
        #df.append(df2,ignore_index=True)
        #with open('D:\dissertation\dataset\citation-network1\outputacm.csv', 'a') as f:
                #df.to_csv(f, header=False)

# opening the csv file in 'a+' mode
#print(df)
file = open('D:\dissertation\dataset\DBLP-citation-network-Oct-19 (1)\DBLP-citation-network-Oct-19.csv', mode='a+', encoding='utf-8', newline='')
# writing the data into the file
with file:
        write = csv.writer(file)
        write.writerows(final)