# -*- coding: utf-8 -*-
from scipy.integrate import dblquad
from scipy.special import jv
from scipy import integrate
from numpy import vectorize
from numpy import inf
from scipy.integrate import quad
from sympy import init_printing, symbols, integrate
import numpy as np
import sympy
import matplotlib.pyplot as plt

init_printing()
# quad 函数
# 贝塞尔函数


def f(x):
    return jv(2.5, x)


x = np.linspace(0, 10)
plt.plot(x, f(x), 'k-')
plt.show()

interval = [0, 6.5]
value, max_err = quad(f, *interval)
print("integral = {:.9f}".format(value))
print("upper bound on error: {:.2e}".format(max_err))
x = np.linspace(0, 10, 100)
p = plt.plot(x, f(x), 'k-')
x = np.linspace(0, 6.5, 45)
p = plt.fill_between(x, f(x), where=f(x) > 0, color="blue")
p = plt.fill_between(x, f(x), where=f(x) < 0, color="red", interpolate=True)

# 积分到无穷

interval = [0., inf]


def g(x):
    return np.exp(-x ** 1 / 2)


value, max_err = quad(g, *interval)
x = np.linspace(0, 10, 50)
fig = plt.figure(figsize=(10, 3))
p = plt.plot(x, g(x), 'k-')
p = plt.fill_between(x, g(x))
plt.annotate(r"$\int_0^{\infty}e^{-x^1/2}dx = $" + "{}".format(value), (4, 0.6),
             fontsize=16)
print("upper bound on error: {:.1e}".format(max_err))


# 双重积分

def h(x, t, n):
    """core function, takes x, t, n"""
    return np.exp(-x * t) / (t ** n)


@vectorize
def int_h_dx(t, n):
    """Time integrand of h(x)."""
    return quad(h, 0, np.inf, args=(t, n))[0]


@vectorize
def I_n(n):
    return quad(int_h_dx, 1, np.inf, args=(n))


print(I_n([0.5, 1.0, 2.0, 5]))

# 双重积分
@vectorize
def I(n):
    """Same as I_n, but using the built-in dblquad"""
    x_lower = 0
    x_upper = np.inf
    return dblquad(h, lambda t_lower: 1, lambda t_upper: np.inf,
                   x_lower, x_upper, args=(n,))


print(I_n([0.5, 1.0, 2.0, 5]))

"""
使用 np.ufunc 进行积分
采样点积分 trapz 方法 和 simps 方法

"""

from scipy import integrate
def half_circle(x):
    return (1-x**2)**0.5


N = 10000
x = np.linspace(-1, 1, N)
dx = 2.0/N
y = half_circle(x)
# 面积的两倍
print(dx * np.sum(y[:-1] + y[1:]))
# 面积的两倍
print(np.trapz(y, x) * 2)
pi_half, err = integrate.quad(half_circle, -1, 1)
print(pi_half * 2)
