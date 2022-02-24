import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from scipy import linalg
"""不建议使用
A = np.mat("[1, 2; 3, 4]")
print (repr(A))
print (repr(A.I))# .I 表示逆，
print (repr(A.T))# .T 表示转置

b = np.mat('[5; 6]')
print (repr(A * b)) #矩阵乘法
"""
A = np.array([[1, 2], [3, 4]])
print(repr(A))
# .I 表示逆，
print(linalg.inv(A))
# .T 表示转置
print(repr(A.T))
b = np.array([5, 6])
# 矩阵乘法
print(repr(A.dot(b)))
# 普通乘法
print(repr(A * b))
# 求逆
A = np.array([[1, 2], [3, 4]])
print(linalg.inv(A))
print(A.dot(scipy.linalg.inv(A)))

A = np.array([[1, 3, 5],
              [2, 5, 1],
              [2, 3, 8]])
b = np.array([10, 8, 3])
x = linalg.solve(A, b)
print(x)

A = np.array([[1, 3, 5],
              [2, 5, 1],
              [2, 3, 8]])
# 计算行列式
print(linalg.det(A))

# http://nbviewer.jupyter.org/github/lijin-THU/notes-python/blob/master/04-scipy/04.09-linear-algbra.ipynb
