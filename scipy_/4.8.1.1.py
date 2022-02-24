import numpy as np
import sympy

x = sympy.symbols('x')
i = sympy.exp(-x**2)
I = sympy.integrate(i,  (x, 0, 1))
print("sympy: {:.6f}".format(I.evalf()))

from scipy import integrate


def f(x):
    return np.exp(-x**2)


interval = [0, 1]
value1, max_err = integrate.quad(f, *interval)
print("scipy1: {:.6f}".format(value1))
value2 = integrate.quad(f, 0, 1)
print("scipy2: {:.6f}".format(value2[0]))
