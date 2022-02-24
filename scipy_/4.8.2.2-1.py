import numpy as np
import sympy


x, y = sympy.symbols('x y')
s = x**2+y**2
s1 = sympy.integrate(s,  x)
s2 = s1.subs(x, 2)-s1.subs(x, sympy.sqrt(y))
s3 = sympy.integrate(s2,  (y, 1, 4))
print("sympy: {:.6f}".format(s3.evalf()))


from scipy import integrate

s = integrate.dblquad(lambda x, y: x**2+y**2, 1, 4,
                      lambda x: np.sqrt(x), lambda x: 2)
print("scipy: {:.6f}".format(s[0]))
