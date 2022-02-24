# -*- coding: utf-8 -*-
#

import numpy as np
from scipy.integrate import odeint
import pylab as pl


def set_ch():
    # 指定默认字体
    pl.rcParams['font.sans-serif'] = ['FangSong']
    # 解决保存图像是负号'-'显示为方块的问题
    pl.rcParams['axes.unicode_minus'] = False


def fun1(y, t):
    a = -2.0
    b = -0.1
    return np.array([y[1], a*y[0]+b*y[1]])


set_ch()
t = np.arange(0.0, 50.0, 1000)
track = odeint(fun1, np.array([0.0005, 0.2]), t)
pl.plot(t, track[:, 0])
pl.show()
