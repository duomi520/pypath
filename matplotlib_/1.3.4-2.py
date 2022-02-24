import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
x = np.arange(-8, 8, 0.5)
x = np.ones((32, 32))*x
y = np.arange(-8, 8, 0.5)
y = y*np.ones((32, 32))
y = np.transpose(y)
r = np.sqrt(x**2+y**2)+1e-10
z = np.sin(r)/r
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z, cmap=plt.get_cmap("jet"))
plt.show()
