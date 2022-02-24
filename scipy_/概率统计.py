# -*- coding: utf-8 -*-
from scipy.stats import norm
import numpy as np
import scipy.stats.stats as st
import pylab
heights = np.array([1.46, 1.79, 2.01, 1.75, 1.56,
                    1.69, 1.88, 1.76, 1.88, 1.78])
# 忽略nan值之后的中位数
print('median, ', np.nanmedian(heights))
# 众数及其出现次数
print('mode, ', st.mode(heights))
# 偏度
print('skewness, ', st.skew(heights))
# 峰度
print('kurtosis, ', st.kurtosis(heights))
print('and so many more...')

# 正态分布
x_norm = norm.rvs(size=500)
h = pylab.hist(x_norm, normed=True, bins=20)
print('counts, ', h[0])
print('bin centers', h[1])

x_mean, x_std = norm.fit(x_norm)

print('mean, ', x_mean)
print('x_std, ', x_std)
x = np.linspace(-3, 3, 50)
p = pylab.plot(x, norm.pdf(x), 'r-')
pylab.show()
