# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:14:27 2017

@author: swwang
"""

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt

from numpy import polyfit, poly1d

x = np.linspace(-3*np.pi, 3*np.pi, 100)
y = np.sin(x)
y1 = poly1d(polyfit(x, y, 1))
y3 = poly1d(polyfit(x, y, 3))
y5 = poly1d(polyfit(x, y, 5))
y7 = poly1d(polyfit(x, y, 7))
y9 = poly1d(polyfit(x, y, 9))
a = plt.axis([-3 * np.pi, 3 * np.pi, -1.25, 1.25])
p = plt.plot(x, np.sin(x), 'k')
p = plt.plot(x, y1(x))
p = plt.plot(x, y3(x))
p = plt.plot(x, y5(x))
p = plt.plot(x, y7(x))
p = plt.plot(x, y9(x))
plt.show()
