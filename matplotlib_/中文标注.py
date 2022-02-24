# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:41:31 2017

@author: swwang
"""

import numpy as np
import matplotlib.pyplot as plt
# from pylab import *


def set_ch():
    from pylab import mpl
    # 指定默认字体
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False


set_ch()
n = 256
X = np.linspace(-np.pi, np.pi, n, endpoint=True)
Y = np.sin(2*X)

plt.plot(X, Y+1, color='blue', alpha=1.00)
plt.plot(X, Y-1, color='red', alpha=1.00)

plt.annotate('发货', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.show()
