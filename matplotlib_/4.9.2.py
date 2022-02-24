import numpy as np
import pylab
from scipy.stats import norm

x = np.random.randn(100)
pylab.figure(1)
pylab.subplot(221)
pylab.hist(x, bins=7)
pylab.subplot(222)
pylab.hist(x, bins=20)
t = np.linspace(-2,2,100)
p = pylab.plot(t, norm.pdf(t)*30, 'r')

y = np.random.rand(100)
y1 = np.arange(min(y), max(y), 0.01)
y2 = np.arange(min(y), max(y), 0.05)
pylab.subplot(223)
pylab.hist(y1, bins=10)
pylab.subplot(224)
pylab.hist(y2, bins=20)

pylab.show()
