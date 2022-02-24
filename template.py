# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

# 显示中文支持


def set_ch():
    from pylab import mpl
    # 指定默认字体
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False


set_ch()

# https://www.scipy.org/docs.html
# http://docs.sympy.org/dev/index.html
# https://docs.scipy.org/doc/scipy/reference/tutorial/index.html
# http://cwiki.apachecn.org/spacedirectory/view.action
# https://matplotlib.org/


# https://www.kancloud.cn/wizardforcel/scipy-lecture-notes/129875   数学优化：找到函数的最优解
# https://blog.csdn.net/pipisorry/article/details/51106570          皮皮blog

# http://www.aboutyun.com/thread-23911-1-1.html                     TensorFlow
# http://keras-cn.readthedocs.io/en/latest/                         keras
# https://blog.csdn.net/marsjhao/article/details/67042392           keras
# https://blog.csdn.net/googler_offer/article/details/78726571      keras
# https://blog.csdn.net/ljp1919                                     keras


# 在求解最优化问题中，拉格朗日乘子法（Lagrange Multiplier）和KKT（Karush Kuhn Tucker）条件是两种最常用的方法。在有等式约束时使用拉格朗日乘子法，在有不等约束时使用KKT条件。
