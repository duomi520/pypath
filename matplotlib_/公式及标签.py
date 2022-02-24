# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:14:27 2017

@author: swwang
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy import init_printing
init_printing()
from sympy import symbols, integrate
import sympy


def dy_dt(y, t):
    return np.sin(t)


from scipy.integrate import odeint
t = np.linspace(0, 2*np.pi, 100)
result = odeint(dy_dt, 0, t)
fig = plt.figure(figsize=(12, 4))
p = plt.plot(t, result, "rx", label=r"$\int_{0}^{x}sin(t) dt $")
p = plt.plot(t, -np.cos(t) + np.cos(0), label=r"$cos(0) - cos(t)$")
p = plt.plot(t, dy_dt(0, t), "g-", label=r"$\frac{dy}{dt}(t)$")
l = plt.legend(loc="upper right")
xl = plt.xlabel("t")
plt.show()
