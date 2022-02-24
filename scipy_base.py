from scipy.stats import uniform
from scipy.stats import norm
from scipy.optimize import basinhopping
from scipy.optimize import brute
from scipy.optimize import minimize, rosen
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy import linalg
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from numpy import polyfit
import numpy as np
from math import sqrt, sin
from scipy.integrate import quad, dblquad, tplquad, odeint


"""
子模块           描述
cluster         聚类算法
constants       物理数学常数
fftpack         快速傅里叶变换
integrate       积分和常微分方程求解
interpolate     插值
io              输入输出
linalg          线性代数
odr             正交距离回归
optimize        优化和求根
signal          信号处理
sparse          稀疏矩阵
spatial         空间数据结构和算法
special         特殊方程
stats           统计分布和函数
weave           C/C++ 积分
"""
"""
quad	单积分
dblquad	二重积分
tplquad	三重积分
nquad	n倍多重积分
fixed_quad	高斯积分，阶数n
quadrature	高斯正交到容差
romberg	Romberg积分
trapz	梯形规则
cumtrapz	梯形法则累计计算积分
simps	辛普森的规则
romb	Romberg积分
polyint	分析多项式积分(NumPy)
poly1d	辅助函数polyint(NumPy)
"""
# 单积分  (0.7468241328124271, 8.291413475940725e-15)
print("01\t", quad(lambda x: np.exp(-x**2), 0, 1))
# 双重积分  (0.5, 1.7092350012594845e-14)
print("02\t", dblquad(lambda x, y: 16*x*y, 0, 0.5, lambda x: 0,
                      lambda y: sqrt(1-4*y**2)))
# 三重积分  (224.92153573331143, 1.7753629738496716e-11)
print("03\t", tplquad(lambda x, y, z: x**2+y**2+z**2, 1, 2, lambda x: sqrt(x), lambda x: x**2, lambda x, y: sqrt(x*y),
                      lambda x, y: x**2*y))
# odeint() 是使用LSODA（Livermore Solver for Ordinary Differential equations with Automatic method switching for stiff and non-stiff problems）的通用积分器
# odeint解决如下形式的第一顺序ODE系统：$dy/dt = rhs(y1, y2, .., t0,...)$


def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt


y0 = [np.pi - 0.1, 0.0]
t = np.linspace(0, 10, 101)
sol = odeint(pend, y0, t, args=(.25, 5.))
print("04\t")
# polyint
a = np.array([1.0, 0, -2, 1])
# f(x) =x^3-2x+1  多项式可进行加减乘除运算
p = np.poly1d(a)
# [ 1.        0.515625  0.125    -0.078125  0.      ]
print("05\t", p(np.linspace(0, 1, 5)))
# 微分
print("06\t", p.deriv())
# 积分
print("07\t",  p.integ())
# 多项式的根 [-1.61803399  1.          0.61803399]
print("08\t", np.roots(p))
# 多项式拟合 [-9.22671371e-02  3.94510210e-17  8.52213827e-01 -8.78358292e-17]
x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
y3 = polyfit(x, y, 3)
print("09\t", y3)
# 一维插值  0.5734566736497806
""" kind 参数
nearest 最近邻插值
zero 0阶插值
linear 线性插值
quadratic 二次插值
cubic 三次插值
4,5,6,7 更高阶插值
"""
x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)
f = interp1d(x, y, kind='linear')
print("10\t", f(2.0))
# 样条曲线  0.5991587086521026
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)
f = UnivariateSpline(x, y)
print("11\t", f(0.0))
# 线性方程组    [-9.28  5.16  0.76]
"""
x+3y+5z=10
2x+5y+z=8
2x+3y+8z=3
"""
a = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
b = np.array([10, 8, 3])
print("20\t", linalg.solve(a, b))
# 优化算法


def func_min(x): return x**2 + 10*np.sin(x)


# 曲线拟合
xdata = np.linspace(-10, 10, num=20)
ydata = func_min(xdata) + np.random.randn(xdata.size)
params, params_covariance = curve_fit(
    lambda x, a, b: a*x**2 + b*np.sin(x), xdata, ydata, [2, 2])
# [ 0.9968722  10.20670301] [[2.91129870e-05 4.90495659e-12] [4.90495659e-12 1.48686433e-01]]
print("25\t", params, params_covariance)
# 最小二乘    求解一个带有变量边界的非线性最小二乘问题。 给定残差f(x)(n个实变量的m维实函数)和损失函数rho(s)(标量函数)，最小二乘法找到代价函数F(x)的局部最小值
res = least_squares(lambda x: np.array(
    [10 * (x[1] - x[0]**2), (1 - x[0])]), np.array([2, 2]))
print("26\t", res.x)  # [1. 1.]
# 定点求解:
# 方程组：root()函数可以找到一组非线性方程的根
sol = root(lambda x:  x*2 + 2 * np.cos(x), 0.3)
# [-0.73908513]
print("27\t", sol.x)
# 看图在1附近,算出[0.]
print("28\t", root(func_min, 1).x)
# 看图另一个在-2.5附近,算出 [-2.47948183]
print("29\t", root(func_min, -2.5).x)
# 非线性方程组求解


def fun_fsolve(x):
    x0, x1, x2 = x.tolist()
    return [5*x1+3, 4*x0*x0 - 2*sin(x1*x2), x1*x2-1.5]


# [-0.70622057 -0.6        -2.5       ] [0.0, -9.126033262418787e-14, 5.329070518200751e-15]
print("30\t", fsolve(fun_fsolve, [1, 1, 1]),
      fun_fsolve(fsolve(fun_fsolve, [1, 1, 1])))

# 线性规划求最大值或最小值
# max z=2x1+3x2-5x3
# s.t. x1+x2+x3=7
#      2x1-5x2+x3>=10
#      x1+3x2+x3<=12
#      x1,x2,x3>=0
c = np.array([2, 3, -5])*-1
a = np.array([[-2, 5, -1], [1, 3, 1]])
b = np.array([-10, 12])
# -14.571428571428571
print("31\t", linprog(c, a, b, [[1, 1, 1]], [
      7], bounds=((0, 7), (0, 7), (0, 7))).fun)
"""
使用各种算法(例如BFGS，Nelder-Mead单纯形，牛顿共轭梯度，COBYLA或SLSQP)的无约束和约束最小化多元标量函数(minimize())
全局(蛮力)优化程序(例如，anneal()，basinhopping())
最小二乘最小化(leastsq())和曲线拟合(curve_fit())算法
标量单变量函数最小化(minim_scalar())和根查找(newton())
使用多种算法(例如，Powell，Levenberg-Marquardt混合或Newton-Krylov等大规模方法)的多元方程系统求解(root)
"""
# 单元标量函数最小化
# 1.2807764040333458
res = minimize_scalar(lambda x: (x-2)*x*(x+2)**2)
print("35\t", res.x)
# -2.000000202597239
res = minimize_scalar(lambda x: (x-2)*x*(x+2)**2,
                      bounds=(-3, -1), method='bounded')
print("36\t", res.x)
# 多元标量函数最小化
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead')
# [0.99910115 0.99820923 0.99646346 0.99297555 0.98600385]
print("37\t", res.x)
res = minimize(rosen, x0, method='BFGS')
# [0.99999925 0.99999852 0.99999706 0.99999416 0.99998833]
print("38\t", res.x)
res = minimize(rosen, x0, method='CG')
# [0.99999826 0.99999652 0.99999303 0.99998604 0.99997204]
print("39\t", res.x)
# 全局最优化算法
# 暴力求解
# [-1.30641113]
print("45\t", brute(func_min, ((-10, 10, 0.1),)))
# 模拟退火
# [-1.30644001]
print("46\t", basinhopping(func_min, [1.]).x)
"""
全局优化的一些有用的包
Pyopt:http://www.pyopt.org/index.html
IPOPT:https://github.com/xuy/pyipopt
PyGMO:http://esa.github.io/pygmo/    
PyEvolve:http://pyevolve.sourceforge.net/
"""
# 统计
"""
常见的连续概率分布有：
均匀分布
正态分布
学生t分布
F分布
Gamma分布
离散概率分布：
伯努利分布
几何分布
"""
# 正态连续随机变量

# [0.84134475 0.15865525 0.5 0.84134475 0.9986501  0.99996833  0.02275013 1.]
print("51\t", norm.cdf(np.array([1, -1., 0, 1, 3, 4, -2, 6])))
# 生成随机变量序列
print("52\t", norm.rvs(size=5))

# 均匀分布
print("53\t", uniform.cdf([0, 1, 2, 3, 4, 5], loc=1, scale=4))

# 描述性统计
"""
1	describe()	计算传递数组的几个描述性统计信息
2	gmean()	计算沿指定轴的几何平均值
3	hmean()	计算沿指定轴的谐波平均值
4	kurtosis()	计算峰度
5	mode()	返回模态值
6	skew()	测试数据的偏斜度
7	f_oneway()	执行单向方差分析
8	iqr()	计算沿指定轴的数据的四分位数范围
9	zscore()	计算样本中每个值相对于样本均值和标准偏差的z值
10	sem()	计算输入数组中值的标准误差(或测量标准误差)
"""

# https://wizardforcel.gitbooks.io/scipy-lecture-notes/content/12.html
