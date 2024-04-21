# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as colors
from sklearn.cluster import KMeans


from mpl_toolkits.mplot3d import Axes3D

# importing boston housing dataset from sklearn

from sklearn.datasets import load_boston

# imports train_test_split to divide the dataset into two parts, one the training set and the other, a test set
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

df = pd.read_csv(r"D:\dissertation\S1File.csv")
#print(df['IFBSCP'])
df.loc[df['IFBSCP']=='#DIV/0!', ['IFBSCP']] = 0

col=['No. of cit. preceding 5 years','No. of self-cit. preceding 5 years','IFBSCP']

df=df.loc[:,col]
df.to_csv(r"D:\dissertation\S1File2.csv",index=False)
Y=df.loc[:,['IFBSCP']]
X=df.loc[:,['No. of cit. preceding 5 years','No. of self-cit. preceding 5 years']]
X_train, X_test, y_train, y_test = train_test_split(X,Y)


clf = LinearRegression()

# training the model using the feature and label

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)


expected = y_test


# plotting the best-fit line

plt.figure(figsize=(4, 3))

plt.scatter(expected, predicted)

plt.plot([0, 50], [0, 50], '--k')

plt.axis('tight')

plt.xlabel('True price ($1000s)')

plt.ylabel('Predicted price ($1000s)')

plt.tight_layout()

