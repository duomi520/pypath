# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:37:57 2017

@author: swwang
"""
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

group = np.array([[1.0, 0.9], [1.0, 1.0], [1.1, 1.0], [0.1, 0.2], [0.0, 0.1], [0.1, 0.1]])  
labels = ['A','A', 'A', 'B', 'B', 'B'] 
testX = np.array([1.2, 1.0]) 
knn = KNeighborsClassifier()
knn.fit(group, labels) 
print ("Your input is:", testX, "and classified to class: ", knn.predict(testX)) 
testX = np.array([0.1, 0.3]) 
print ("Your input is:", testX, "and classified to class: ", knn.predict(testX)) 