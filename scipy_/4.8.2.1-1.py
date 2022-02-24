import numpy as np
import sympy

x, y = sympy.symbols('x y')
s = x**y
s1 = sympy.integrate(s,  x)
s2 = s1.subs(x, 1)-s1.subs(x, 0)
s3 = sympy.integrate(s2,  (y, 1, 2))
print("sympy: {:.6f}".format(s3.evalf()))

from scipy import integrate

s = integrate.dblquad(lambda x, y: x**y, 1, 2, lambda x: 0, lambda x: 1)
print("scipy: {:.6f}".format(s[0]))
