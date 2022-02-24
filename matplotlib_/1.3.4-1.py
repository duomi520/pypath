import matplotlib.pyplot as plt
import numpy as np
t = np.arange(0.0, 4*np.pi, np.pi/50)
y0 = np.exp(-t/3)
y = np.exp(-t/3)*np.sin(3*t)
plt.grid()
plt.axis([0, 14, -1, 1])
plt.plot(t, y, color='red', alpha=1.00)
plt.plot(t, y0, color='blue', linestyle='dashed', alpha=1.00)
plt.plot(t, -y0, color='blue', linestyle='dashed', alpha=1.00)
plt.show()
