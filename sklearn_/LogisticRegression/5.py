# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:29:01 2017

@author: swwang
"""
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model

## step 1: load data  
print ("step 1: load data...")  
#train_x, train_y = loadData()  
#test_x = train_x; test_y = train_y  
t_train_x = []  
t_train_y = [] 
fileIn = open('D:/PyPath/sklearn_/LogisticRegression/testSet.txt')  
for line in fileIn.readlines():  
    lineArr = line.strip().split()  
    t_train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])  
    t_train_y.append(float(lineArr[2])) 
    
train_x=np.mat(t_train_x)
train_y=np.mat(t_train_y).transpose()
## step 2: training...  
print ("step 2: training..." ) 
logreg = linear_model.LogisticRegression()
logreg.fit(train_x,train_y)
## step 3: testing  
print ("step 3: testing...")
cc=logreg.predict(train_x)
## step 4: show the result  
print ("step 4: show the result...")
cy=  np.array(t_train_y)
accuracy=((cy-cc)==0)
print ('The classify accuracy is: %.3f%%' % (np.mean(accuracy) * 100) ) 

numSamples, numFeatures = np.shape(train_x)  
# draw all samples  
for i in range(numSamples):  
    if int(train_y[i, 0]) == 0:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')  
    elif int(train_y[i, 0]) == 1:  
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')  
  
plt.xlabel('X1'); plt.ylabel('X2')  
plt.show()          