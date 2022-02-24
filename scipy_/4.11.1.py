import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

t = np.linspace(0, 5, 100)


def y_func(tt):
    return 1-np.cos(3*tt)*np.exp(-tt)


y = y_func(t)
plt.plot(t, y, 'b')
plt.grid()
plt.plot(t, 0.95*np.ones(np.size(t)), 'r')
plt.show()

t_nearest = interp1d(y[8:14], t[8:14], kind='nearest')
t_linear = interp1d(y[8:14], t[8:14], kind='linear')
t_cubic = interp1d(y[8:14], t[8:14], kind='cubic')
t_slinear = interp1d(y[8:14], t[8:14], kind='slinear')
print("t_nearest:{:.4f}".format(t_nearest(0.95)))
print("t_linear:{:.4f}".format(t_linear(0.95)))
print("t_cubic:{:.4f}".format(t_cubic(0.95)))
print("t_slinear:{:.4f}".format(t_slinear(0.95)))

from scipy.optimize import fsolve
t_zero = fsolve(lambda x: 1-np.cos(3*x)*np.exp(-x)-0.95, .5)
print("t_zero:{:.4f}".format(t_zero[0]))
