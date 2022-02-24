import matplotlib.pyplot as plt
import numpy as np
a = 0.1
b = 0.5
t = np.arange(-10, 10, 0.01)
y = np.sin(t)**2*np.exp(-a*t)-b*np.abs(t)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.plot(t, y, 'r')
plt.plot(t, np.zeros(np.size(t)), 'k')
plt.show()
