# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:17:52 2017

@author: swwang
"""

import numpy as np
import matplotlib.pyplot as plt

plt.grid(0)

n = 256
X = np.linspace(-np.pi, np.pi, n, endpoint=True)
Y = np.sin(2*X)

plt.plot(X, Y+1, color='blue', alpha=1.00)
plt.plot(X, Y-1, color='red', alpha=1.00)
plt.show()
