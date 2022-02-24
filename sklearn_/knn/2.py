# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:37:57 2017

@author: swwang
"""
import os 
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# convert image to vector  
def  img2vector(filename):  
    rows = 32  
    cols = 32  
    imgVector = np.zeros((1, rows * cols))   
    fileIn = open(filename)  
    for row in range(rows):  
        lineStr = fileIn.readline()  
        for col in range(cols):  
            imgVector[0, row * 32 + col] = int(lineStr[col])  
    return imgVector  
  
# load dataSet  
def loadDataSet():  
    print ("---Getting training set...") 
    dataSetDir = 'D:/PyPath/sklearn_/knn/'  
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits') # load the training set  
    numSamples = len(trainingFileList)  
    train_x = np.zeros((numSamples, 1024))  
    train_y = []  
    for i in range(numSamples):  
        filename = trainingFileList[i]  
        # get train_x  
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename)   
        # get label from file name such as "1_18.txt"  
        label = int(filename.split('_')[0]) # return 1  
        train_y.append(label)  
    print ("---Getting testing set...")  
    testingFileList = os.listdir(dataSetDir + 'testDigits') # load the testing set  
    numSamples = len(testingFileList)  
    test_x = np.zeros((numSamples, 1024))  
    test_y = []   
    for i in range(numSamples):  
        filename = testingFileList[i]  
        # get train_x  
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename)   
         # get label from file name such as "1_18.txt"  
        label = int(filename.split('_')[0]) # return 1  
        test_y.append(label) 
    return train_x, train_y, test_x, test_y  
 
x,y,tx,ty=loadDataSet()
knn = KNeighborsClassifier()
knn.fit(x, y) 
num=len(ty)
matchCount = 0 
for row in range(num): 
    if ty[row]==knn.predict(np.array(tx[row]).reshape(1, -1)):
        matchCount+=1
accuracy = float(matchCount) / num  
print ("---Show the result..." ) 
print ('The classify accuracy is: %.2f%%' % (accuracy * 100))          
  


