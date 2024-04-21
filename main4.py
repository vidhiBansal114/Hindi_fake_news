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
df = pd.read_csv(r"D:\dissertation\S2.csv")

k='% Journals with IFBSCP > 3.0'
plt.plot(df['Year'], df['% Journals with IFBSCP > 3.0'])
#plt.plot(df['Year'], df['% Journals with IFBSCP > 1.5'],label = '% Journals with IFBSCP > 1.5')
#plt.plot(df['Year'], df['% Journals with IFBSCP > 2.0'],label = '% Journals with IFBSCP > 2.0')
#plt.plot(df['Year'], df['% Journals with IFBSCP > 3.0'],label = '% Journals with IFBSCP > 3.0')
#plt.plot(df['Year'], df['Mean IFBSCP'],label = 'Mean IFBSCP')
# naming the x axis
#plt.yticks([10,20,30,40,50,60,70])
plt.xlabel('Year')
# naming the y axis
plt.ylabel(k)
# giving a title to my graph
plt.title(k)

# show a legend on the plot
plt.legend()

# function to show the plot
plt.savefig('D:\dissertation\Journals with IFBSCP 3.0.png')
print(df)