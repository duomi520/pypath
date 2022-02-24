import numpy as np
import scipy.stats as ss
import pylab

mu = 3
sigma = 0.5
x = 3+sigma*np.array([-3, -2, -1, 1, 2, 3])
yf = ss.norm.cdf(x, mu, sigma)
P = np.array([yf[3]-yf[2], yf[4]-yf[1], yf[5]-yf[0]])
xd = np.arange(1, 5, 0.1)
yd = ss.norm.pdf(xd, mu, sigma)
xx = [0, 0, 0]
yy = [0, 0, 0]
for k in range(3):
    xx[k] = np.arange(x[3-k-1], x[3+k]+0.05, sigma/10)
    yy[k] = ss.norm.pdf(xx[k], mu, sigma)

pylab.figure(1)
pylab.subplot(131)
pylab.plot(xd, yd, 'b')
t1 = np.array([x[2]])
t1 = np.append(t1, xx[0])
t1 = np.append(t1, x[3])
t2 = np.array([0])
t2 = np.append(t2, yy[0])
t2 = np.append(t2, 0)
pylab.fill(t1, t2, 'g')
pylab.text(mu-0.5*sigma, 0.3, '{:.4f}'.format(P[0]))
pylab.subplot(132)
pylab.plot(xd, yd, 'b')
t3 = np.array(x[1])
t3 = np.append(t3, xx[1])
t3 = np.append(t3, x[4])
t4 = np.array([0])
t4 = np.append(t4, yy[1])
t4 = np.append(t4, 0)
pylab.fill(t3, t4, 'g')
pylab.text(mu-0.5*sigma, 0.3, '{:.4f}'.format(P[1]))
pylab.subplot(133)
pylab.plot(xd, yd, 'b')
t5 = np.array([x[0]])
t5 = np.append(t5, xx[2])
t5 = np.append(t5, x[5])
t6 = np.array([0])
t6 = np.append(t6, yy[2])
t6 = np.append(t6, 0)
pylab.fill(t5, t6, 'g')
pylab.text(mu-0.5*sigma, 0.3, '{:.4f}'.format(P[2]))
pylab.show()
