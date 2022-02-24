# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:14:27 2017

@author: swwang
"""

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from numpy import polyfit, poly1d

x = np.linspace(-5, 5, 100)
y = 4 * x + 1.5
noise_y = y + np.random.randn(y.shape[-1]) * 2.5
coeff = polyfit(x, noise_y, 1)

p = plt.plot(x, noise_y, 'rx')
p = plt.plot(x, coeff[0] * x + coeff[1], 'k-')
p = plt.plot(x, y, 'b--')

f = poly1d(coeff)
p = plt.plot(x, f(x), 'g')

plt.show()
