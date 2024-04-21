# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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
from matplotlib.widgets import Slider
# Loading dataset
df = pd.read_csv(r"D:\dissertation\S1File.csv")
#print(df['IFBSCP'])
df.loc[df['IFBSCP']=='#DIV/0!', ['IFBSCP']] = 0
#col=['Year','Journal ID','No. of cit. past 2 years','No. of self-cit. past 2 years']
col=['Year','Journal ID','diff','No. of cit. past 2 years','No. of self-cit. past 2 years','No. of cit. preceding 5 years','No. of self-cit. preceding 5 years','IFBSCP']
#print(df['IFBSCP'])
df=df.loc[:,col]
#kmeans = KMeans(2)
#kmeans.fit(df)
#identified_clusters = kmeans.fit_predict(df)
#sns.pairplot(df)
# create a figure and axis
name=[]
past2=[]
past2self=[]
past5=[]
past5self=[]
c=0
#plt.scatter(df["Journal ID"], df["diff"])
#print(df['diff'])
q_low = df["No. of cit. past 2 years"].quantile(0.01)
q_hi  = df["No. of cit. past 2 years"].quantile(0.99)

diff=[]
name=[]
#df_filtered = df[(df["No. of cit. past 2 years"] < q_hi)]
for ind in df.index:
        if df['Year'][ind]==2015 and df['No. of cit. preceding 5 years'][ind]!=0 :
                #if df['No. of self-cit. past 2 years'][ind]!=0 and (df['No. of self-cit. past 2 years'][ind] / df['No. of cit. past 2 years'][ind])>1.:
                        name.append(df['Journal ID'][ind])
                        #past2.append(df['No. of self-cit. past 5 years'][ind])
                        #diff.append(df['diff'][ind])
                        #past2self.append(df['No. of self-cit. past 5 years'][ind])
                        past5.append(df['No. of cit. past 2 years'][ind])
                        past5self.append(df['No. of self-cit. past 2 years'][ind]/df['No. of cit. past 2 years'][ind])
                        c=c+1
print(c)
     #print(df['No. of cit. past 2 years'][ind], df['No. of self-cit. past 2 years'][ind])
from pandas.plotting import scatter_matrix
#q = df["No. of cit. past 2 years"].quantile(0.99)
#df.plot( kind='bar', stacked=True,        title='Stacked Bar Graph by dataframe')
percent2=[]
percent5=[]
final_n=[]
percentDiff=[]
'''
for i in range(c):
        if past5[i]!=0:
                #percentDiff.append(diff[i])
                percent2.append(past5self[i]/past5[i])

                #percentDiff.append((past5self[i]/past5[i])-(past2self[i]/past2[i]))
                final_n.append(name)

'''
plt.scatter(past5,past5self)
#Plot, Axis = plt.subplots()
#plt.subplots_adjust(bottom=0.25)
#plt.bar(final_n,percent, color='r')
# Choose the Slider color
slider_color = 'White'
#plt.plot(final_n,percent)
plt.ylabel('Self citation/Total citation')
#plt.xlabel('Total citation')
# Set the axis and slider position in the plot
#axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],                         facecolor=slider_color)
#slider_position = Slider(axis_position,                         'Pos', 0.1, 90.0)
plt.title('self citation rate for preceeding 2 years\n(calculated for the year 2015)')

# update() function to change the graph when the
# slider is in use
#def update(val):
  #      pos = slider_position.val
  #      Axis.axis([pos, pos + 10, -1, 1])
  #      Plot.canvas.draw_idle()


# update function called using on_changed() function
#slider_position.on_changed(update)
#plt.tick_params(    axis='x',       which='both',bottom=False,    top=False,       labelbottom=False)
#plt.bar(name,past2self, bottom=past2self, color='b')
plt.savefig('D:\dissertation\self citation rate for preceeding 2 years (calculated for the year 2015)')
#plt.show()
#fig, ax = plt.subplots(figsize=(12,12))
#scatter_matrix(df, alpha=1, ax=ax)
#plt.scatter(df['No. of cit. past 2 years'], df['No. of cit. preceding 5 years'])
#plt.show()