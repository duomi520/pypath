# -*- coding: utf-8 -*-
# 径向基函数插值
from scipy.interpolate.rbf import Rbf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d

# 设置 Numpy 浮点数显示格式
np.set_printoptions(precision=3, suppress=True)
data = np.genfromtxt("C-067.txt",
                     delimiter="\t",  # TAB 分隔
                     skip_header=1,  # 忽略首行
                     names=True,  # 读入属性
                     missing_values="INFINITE",  # 缺失值
                     filling_values=0  # 填充缺失值
                     )
for row in data[:7]:
    print("{}\t{}".format(row['TK'], row['Cp']))
print("...\t...")
p = plt.plot(data['TK'], data['Cp'], 'kx')
t = plt.title("JANAF data for Methane $CH_4$")
a = plt.axis([0, 6000, 30, 120])
x = plt.xlabel("Temperature (K)")
y = plt.ylabel(r"$C_p$ ($\frac{kJ}{kg K}$)")


ch4_cp = interp1d(data['TK'], data['Cp'],
                  bounds_error=False, fill_value=-999.25)

""" kind 参数
nearest 最近邻插值
zero 0阶插值
linear 线性插值
quadratic 二次插值
cubic 三次插值
4,5,6,7 更高阶插值
"""
print(ch4_cp(382.2))
print(ch4_cp([32.2, 323.2]))


cp_rbf = Rbf(data['TK'], data['Cp'], function="multiquadric")
"""
multiquadric
gaussian
nverse_multiquadric 
"""

p = plt.plot(data['TK'], cp_rbf(data['TK']), 'r-')

'''
# 高维 RBF 插值
from mpl_toolkits.mplot3d import Axes3D

x1, y1 = np.mgrid[-np.pi/2:np.pi/2:5j, -np.pi/2:np.pi/2:5j]
z1 = np.cos(np.sqrt(x1**2 + y1**2))
fig = plt.figure(figsize=(12, 6))
ax = fig.gca(projection="3d")
ax.scatter(x1, y1, z1)
zz = Rbf(x1, y1, z1)
xx, yy = np.mgrid[-np.pi/2:np.pi/2:50j, -np.pi/2:np.pi/2:50j]
fig = plt.figure(figsize=(12, 6))
ax = fig.gca(projection="3d")
ax.plot_surface(xx, yy, zz(xx, yy), rstride=1,
                cstride=1, cmap=cm.get_cmap("jet"))
'''
plt.show()
