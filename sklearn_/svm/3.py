# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:18:11 2017

@author: swwang
"""

import os
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm

print("step 1: load data...")
dataSet = []
labels = []
fileIn = open('D:/PyPath/sklearn_/svm/testSet.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])
    labels.append(float(lineArr[2]))

dataSet = np.mat(dataSet)
labels = np.mat(labels).T
train_x = dataSet[0:80, :]
train_y = labels[0:80, :]
test_x = dataSet[80:100, :]
test_y = labels[80:100, :]
print("step 2: training...")
clf = svm.SVC()
clf.C = 0.6
clf.kernel = 'rbf'
clf.tol = 0.001
clf.fit(train_x, train_y)
print("step 3: testing...")
ty = clf.predict(test_x)
print("step 4: show the result...")
cc = np.asarray(test_y).reshape(20,)
accuracy = ((ty-cc) == 0)
print('The classify accuracy is: %.3f%%' % (np.mean(accuracy) * 100))
# draw all samples
numSamples = train_x.shape[0]
alphas = np.mat(np.zeros((numSamples, 1)))
for i in range(numSamples):
    if train_y[i] == -1:
        plt.plot(train_x[i, 0], train_x[i, 1], 'or')
    elif train_y[i] == 1:
        plt.plot(train_x[i, 0], train_x[i, 1], 'ob')
plt.show()
print(clf)
