# -*- coding: utf-8 -*-
from sympy import init_printing, symbols, integrate
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

init_printing()


def dy_dt(y, t):
    """
    Governing equations for projectile motion with drag.
    y[0] = position
    y[1] = velocity
    g = gravity (m/s2)
    D = drag (1/s) = force/velocity
    m = mass (kg)
    """
    g = -9.8
    D = 0.1
    m = 0.15
    dy1 = g - (D/m) * y[1]
    dy0 = y[1] if y[0] >= 0 else 0.
    return [dy0, dy1]


position_0 = 0.
velocity_0 = 100
t = np.linspace(0, 12, 100)
y = odeint(dy_dt, [position_0, velocity_0], t)
p = plt.plot(t, y[:, 0])
yl = plt.ylabel("Height (m)")
xl = plt.xlabel("Time (s)")
plt.show()
