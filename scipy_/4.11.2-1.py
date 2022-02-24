import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d


x = np.linspace(-5, 5, num=11)
y = np.linspace(-5, 5, num=11)
X, Y = np.meshgrid(x, y)
zz = 1.2*np.exp(-((X-1)**2+(Y-2)**2))-0.7*np.exp(-((X+2)**2+(Y+1)**2))
Z = -500+zz+np.random.randn(np.size(x), np.size(y))*0.05

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, Z, cmap=plt.get_cmap("jet"))

f = interp2d(x, y, Z, kind='cubic')

xi = np.linspace(-5, 5, num=51)
yi = np.linspace(-5, 5, num=51)
XI, YI = np.meshgrid(xi, yi)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(XI, YI, f(xi, yi), cmap=plt.get_cmap("jet"))

plt.show()
