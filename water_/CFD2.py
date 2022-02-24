import numpy as np
import matplotlib.pylab as plt
import time
import sys
# try changing this number from 41 to 81 and Run All ... what happens?
nx = 41
dx = 2 / (nx - 1)
# nt is the number of timesteps we want to calculate
nt = 25
# dt is the amount of time each timestep covers (delta t)
dt = .025
# assume wavespeed of c = 1
c = 1

# 设置纵轴的上下限
plt.ylim(0.8, 2.2)


def fun(x1, x2):
    x = x1 - c * (dt / dx) * x1 * (x1 - x2)
    return x


u = np.ones(nx)
u[int(.5 / dx):int(1 / dx + 1)] = 2
plt.plot(np.linspace(0, 2, nx), u, label="initial")

for t in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = fun(un[i], un[i - 1])

plt.plot(np.linspace(0, 2, nx), u, label="converged")
plt.legend(loc='right')

plt.show()
