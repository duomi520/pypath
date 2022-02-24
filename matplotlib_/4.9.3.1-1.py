import numpy as np
import scipy.stats as ss
import pylab
x = np.arange(0, 50, 1)
xp = ss.poisson(20).pmf(x)
xn = ss.norm.pdf(x, loc=20, scale=np.sqrt(20))
pylab.plot(x, xn, 'b-')
pylab.plot(x, xp, 'r+')
pylab.text(30, 0.07, r'$\mu= \lambda= 20$')
pylab.show()
