# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:45:02 2017

@author: swwang
"""

# 源代码: Gaël Varoquaux
# 修改以进行文档化:Jaques Grobler
# 协议: BSD 3
from sklearn.datasets import load_digits
from matplotlib import cm
import matplotlib.pyplot as plt
# 加载数字数据集
data = load_digits()
# 展示第一个数字
plt.figure(1, figsize=(3, 3))
plt.imshow(data.get("images")[-1],
           cmap=cm.get_cmap("gray_r"), interpolation='nearest')
plt.show()
