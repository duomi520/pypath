import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 1, 0.1)
y = x*np.exp(-x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y=x*exp(-x)')
plt.plot(x, y)
plt.show()
