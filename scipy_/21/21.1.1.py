# -*- coding: utf-8 -*-

from math import sqrt, sin
import numpy as np
from scipy.integrate import odeint
from scipy.special import ellipk
import scipy.optimize as op
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


def pendulum_th(t, l, th0):
    track = odeint(pendulum_equations, (th0, 0), [0, t], args=(l,))
    return track[-1, 0]


def pendulum_period(l, th0):
    t0 = 2*np.pi*np.sqrt(l/g) / 4
    t = op.fsolve(pendulum_th, t0, args=(l, th0))
    return t*4


set_ch()
ths = np.arange(0, np.pi/2.0, 0.01)
periods = [pendulum_period(1, th) for th in ths]
periods2 = 4*sqrt(1.0/g)*ellipk(np.sin(ths/2)**2)
pl.plot(ths, periods2, color='red', linewidth=4)
pl.plot(ths, periods, color='blue', linewidth=2)
pl.show()
