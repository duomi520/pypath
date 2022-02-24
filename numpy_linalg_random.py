import numpy.random as random
import numpy as np
# 矩阵
a = np.array([3, 4])
np.linalg.norm(a)
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
c = np.array([1, 0, 1])
# 矩阵和向量之间的乘法
# array([ 4, 10, 16])
np.dot(b, c)
# array([ 4, 10, 16])
np.dot(c, b.T)
# 求矩阵的迹，15
np.trace(b)
# 求矩阵的行列式值，0
np.linalg.det(b)
# 求矩阵的秩，2，不满秩，因为行与行之间等差
np.linalg.matrix_rank(b)
d = np.array([
    [2, 1],
    [1, 2]
])
'''
对正定矩阵求本征值和本征向量
本征值为u，array([ 3.,  1.])
本征向量构成的二维array为v，
array([[ 0.70710678, -0.70710678],
       [ 0.70710678,  0.70710678]])
是沿着45°方向
eig()是一般情况的本征值分解，对于更常见的对称实数矩阵，
eigh()更快且更稳定，不过输出的值的顺序和eig()是相反的
'''
u, v = np.linalg.eig(d)
# Cholesky分解并重建
l = np.linalg.cholesky(d)
'''
array([[ 2.,  1.],
       [ 1.,  2.]])
'''
np.dot(l, l.T)
e = np.array([
    [1, 2],
    [3, 4]
])
# 对不镇定矩阵，进行SVD分解并重建
U, s, V = np.linalg.svd(e)
S = np.array([
    [s[0], 0],
    [0, s[1]]
])
'''
array([[ 1.,  2.],
       [ 3.,  4.]])
'''
np.dot(U, np.dot(S, V))
# 概率
# 设置随机数种子
random.seed(42)
# 产生一个1x3，[0,1)之间的浮点型随机数
# array([[ 0.37454012,  0.95071431,  0.73199394]])
# 后面的例子就不在注释中给出具体结果了
random.rand(1, 3)
# 产生一个[0,1)之间的浮点型随机数
random.random()
# 下边4个没有区别，都是按照指定大小产生[0,1)之间的浮点型随机数array，不Pythonic…
random.random((3, 3))
random.sample((3, 3))
random.random_sample((3, 3))
random.ranf((3, 3))
# 产生10个[1,6)之间的浮点型随机数
5*random.random(10) + 1
random.uniform(1, 6, 10)
# 产生10个[1,6]之间的整型随机数
random.randint(1, 6, 10)
# 产生2x5的标准正态分布样本
random.normal(size=(5, 2))
# 产生5个，n=5，p=0.5的二项分布样本
random.binomial(n=5, p=0.5, size=5)
a = np.arange(10)
# 从a中有回放的随机采样7个
random.choice(a, 7)
# 从a中无回放的随机采样7个
random.choice(a, 7, replace=False)
# 对a进行乱序并返回一个新的array
b = random.permutation(a)
# 对a进行in-place乱序
random.shuffle(a)
# 生成一个长度为9的随机bytes序列并作为str返回
# '\x96\x9d\xd1?\xe6\x18\xbb\x9a\xec'
random.bytes(9)
