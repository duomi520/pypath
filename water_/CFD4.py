#!/usr/bin/python
# coding = utf-8
# 3D图必须引入这个库
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt

nx = 81
ny = 81
nt = 60
c = 1
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))

u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
"""
for n in range(nt+1):
	un = u.copy()
	for i in range(1,len(u)):
		for j in range(1,len(u)):
			u[i,j] = un[i, j] - (c*dt/dx*(un[i,j] - un[i-1,j]))-(c*dt/dy*(un[i,j]-un[i,j-1]))
			u[0,:] = 1
			u[-1,:] = 1
			u[:,0] = 1
			u[:,-1] = 1
"""
# 下面的与上面的功能一样，只是形式简单

for n in range(nt + 1):
    un = u.copy()
    u[1:, 1:] = un[1:, 1:] - c * \
        (dt / dx) * (un[1:, 1:] - un[0:-1, 1:]) - \
        (dt / dy) * (un[1:, 1:] - un[1:, 0:-1])
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

fig = plt.figure(dpi=100)
ax = plt.subplot(111, projection="3d")
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u[:])

plt.show()
