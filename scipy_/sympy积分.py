# -*- coding: utf-8 -*-
from sympy import init_printing, symbols, integrate
import numpy as np
import sympy
init_printing()

x, y = symbols('x y')
z=sympy.sqrt(x ** 2 + y ** 2)
z.subs(x, 3)

from sympy.abc import theta
y = sympy.sin(theta) ** 2
Y = integrate(y)
print(Y.subs(theta, np.pi) - Y.subs(theta, 0))
print(integrate(y, (theta, 0, sympy.pi)))
Y_indef = sympy.Integral(y)  #不定积分
Y_def = sympy.Integral(y, (theta, 0, sympy.pi)) #定积分

Y_raw = lambda x: integrate(y, (theta, 0, x))
Y = np.vectorize(Y_raw)

import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi)
p = plt.plot(x, Y(x))
t = plt.title(r'$Y(x) = \int_0^x sin^2(\theta) d\theta$')
plt.show()