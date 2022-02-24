import numpy as np
import sympy

x = sympy.symbols('x')
i = sympy.sqrt(sympy.log(1/x))
I = sympy.integrate(i,  (x, 0, 1))
print("sympy: {:.6f}".format(I.evalf()))

from scipy import integrate


def f(x):
    return np.sqrt(np.log(1/x))


interval = [0, 1]
value, max_err = integrate.quad(f, *interval)
print("scipy: {:.6f}".format(value))
