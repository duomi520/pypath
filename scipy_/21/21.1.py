# -*- coding: utf-8 -*-

from math import sin
import numpy as np
from scipy.integrate import odeint
import pylab as pl
g = 9.8


def set_ch():
    # 指定默认字体
    pl.rcParams['font.sans-serif'] = ['FangSong']
    # 解决保存图像是负号'-'显示为方块的问题
    pl.rcParams['axes.unicode_minus'] = False


def pendulum_equations(w, t, l):
    th, v = w
    dth = v
    dv = - g/l * sin(th)
    return dth, dv


set_ch()
t = np.arange(0, 10, 0.01)
track = odeint(pendulum_equations, (1, 0), t, args=(1,))
pl.plot(t, track[:, 0])
pl.title(u"单摆的角度变化, 初始角度=1.0弧度")
pl.xlabel(u"时间(秒)")
pl.ylabel(u"震度角度(弧度)")
pl.show()
