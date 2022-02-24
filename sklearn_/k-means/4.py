# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:24:04 2017

@author: swwang
"""
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

## step 1: load data  
print ("step 1: load data...")  
dataSet = []  
fileIn = open('D:/PyPath/sklearn_/k-means/testSet.txt')  
for line in fileIn.readlines():  
    lineArr = line.strip().split('\t')  
    dataSet.append([float(lineArr[0]), float(lineArr[1])])  
## step 2: clustering...  
print ("step 2: clustering...")  
dataSet = np.mat(dataSet) 
clf = KMeans(n_clusters=4)
s = clf.fit(dataSet)
y_pred =clf.fit_predict(dataSet)
print (s)
## step 3: show the result  
print ("step 3: show the result..." )
# show your cluster only available with 2-D data  
numSamples, dim = dataSet.shape  
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
# draw all samples  
for i in range(numSamples):  
    plt.plot(dataSet[i, 0], dataSet[i, 1], mark[y_pred[i]])  
 
plt.show()  