# -*- coding: utf-8 -*-
# SymPy 1.1.1
# pylint: disable=W0614
from sympy import *
# basic operations
x, y, z = symbols('x y z')
expr = x + 1
# 替换 3
print("01\t", expr.subs(x, 2))
# 1 + 2*pi
print("02\t", (1 + x*y).subs([(x, pi), (y, 2)]))
str_expr = "x**2 + 3*x - 1/2"
# 字符转公式 x**2 + 3*x - 1/2
print("03\t", sympify(str_expr))
# 计算值，50为小数位数 3.1415926535897932384626433832795028841971693993751
print("04\t", pi.evalf(50))
# 计算值 3.40000000000000
print("05\t", expr.evalf(subs={x: 2.4}))
# 分数 1/3
print("06\t", Rational(1, 3))
# 阶乘n! 120
print("07\t", factorial(5))
# printing
# In the IPython
init_printing()
pprint(Integral(sqrt(1/x), x), use_unicode=False)
"""
 ╱
 |
 | ___
 | ╱ 1
 |╱ - dx
 |╲╱ x
 |
╱
"""
f = Function('f')
pprint(Derivative(f(x, y), x, y), use_unicode=False)
"""
2
d
-----(f(x, y))
dy dx
"""
# LATEX \int \sqrt{\frac{1}{x}}\, dx
print("10\t", latex(Integral(sqrt(1/x), x)))
# simplification
# 代数简化 x - 1
print("21\t", simplify((x**3 + x**2 - x - 1)/(x**2 + 2*x + 1)))
# 代数展开 x**2 - x - 6
print("22\t", expand((x + 2)*(x - 3)))
# 代数分解 z*(x + 2*y)**2
print("23\t", factor(x**2*z + 4*x*y*z + 4*y**2*z))
# 代数合并 x**3 + x**2*(-z + 2) + x*(y + 1) - 3
print("24\t", collect(x*y + x - 3 + 2*x**2 - z*x**2 + x**3, x))
# 代数消除 (x + 1)/x
print("25\t", cancel((x**2 + 2*x + 1)/(x**2 + x)))
# 代数分式分解 -1/(x + 2) + 1/(x + 1)
print("26\t", apart(1/((x+2)*(x+1)), x))
# 三角函数简化 cos(4*x)/2 + 1/2
print("27\t", trigsimp(sin(x)**4 - 2*cos(x)**2*sin(x)**2 + cos(x)**4))
# 三角函数展开 2*sin(x)*cos(x) + 2*cos(x)**2 - 1
print("28\t", expand_trig(sin(2*x) + cos(2*x)))
a, b = symbols('a b', real=True)
# 指数合并 x**(a + b)
print("29\t", powsimp(x**a*x**b))
# 指数展开 x**a*x**b
print("30\t", expand_power_exp(x**(a + b)))
# 指数展开 z**(a*b)
print("31\t", powdenest((z**a)**b, force=True))
# 对数展开 2*log(z)
print("32\t", expand_log(log(z**2), force=True))
# 对数合并 log(x**a)
print("33\t", logcombine(a*log(x), force=True))
# calculus
# 微分 cos(x)
print("40\t", diff(sin(x), x))
# 高阶微分 x**3*y**2*(x**3*y**3*z**3 + 14*x**2*y**2*z**2 + 52*x*y*z + 48)*exp(x*y*z)
print("41\t", diff(exp(x*y*z), x, y, 2, z, 4))
# 偏微分 3*y - 1
print("42\t", diff((3*x*y + 2*y - x), x, 1))
# 初等函数积分 x**6
print("43\t", integrate(6*x**5, x))
# 特殊函数积分 sqrt(pi)*erf(x)**2/4
print("44\t", integrate(exp(-x**2)*erf(x), x))
# 定积分 0
print("45\t", integrate(x**3, (x, -1, 1)))
# 广义积分 1
print("46\t", integrate(exp(-x), (x, 0, oo)))
integ = Integral((x**4 + x**2*exp(x) - x**2 - 2*x*exp(x) - 2 *
                  x - exp(x))*exp(x)/((x - 1)**2*(x + 1)**2*(exp(x) + 1)), x)
# 复杂积分计算 log(exp(x) + 1) + exp(x)/(x**2 - 1)
print("47\t", integ.doit())
# 极限 1
print("48\t", limit(sin(x)/x, x, 0))
# 级数展开 1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 + O(x**10)
print("49\t", cos(x).series(x, 0, 10))
i, n, m = symbols('i n m', integer=True)
# 求和 Sum(log(n)**(-n), (n, 2, oo))
pprint(summation(1/log(n)**n, (n, 2, oo)))
"""
∞
___
╲
╲ -n
╱ log (n)
╱
???
n = 2
"""
# 有限差分法
f = Function('f')
# 微分方程
dfdx = f(x).diff(x)
# -f(x - 1/2) + f(x + 1/2)
print("50\t", dfdx.as_finite_difference())
f, g = symbols('f g', cls=Function)
# -f(x - 1/2)*g(x - 1/2) + f(x + 1/2)*g(x + 1/2)
print("51\t", differentiate_finite(f(x)*g(x)))

# solvers
# 求解1 {-1, 1}
print("60\t", solveset(Eq(x**2 - 1, 0), x))
# 求解2 ImageSet(Lambda(_n, 2*_n*pi + pi/2), S.Integers)
print("61\t", solveset(sin(x) - 1, x, domain=S.Reals))
# 线性方程求解1 {(-y - 1, y, 2)}
print("62\t", linsolve([x + y + z - 1, x + y + 2*z - 3], (x, y, z)))
# 线性方程求解2 {(-y - 1, y, 2)}
print("63\t", linsolve(
    Matrix(([1, 1, 1, 1], [1, 1, 2, 3])), (x, y, z)))
a, b, c, d = symbols('a, b, c, d', real=True)
# 非线性方程求解1 {(-1, -1), (0, 0)}
print("64\t", nonlinsolve([a**2 + a, a - b], [a, b]))
# 非线性方程求解2 {(2, 1/2)}
print("65\t", nonlinsolve([x*y - 1, x - 2], x, y))
# 非线性方程求解3 {(-I, -I), (-I, I), (I, -I), (I, I)}
print("66\t", nonlinsolve([x**2 + 1, y**2 + 1], [x, y]))
print("67\t", roots(x**3 - 6*x**2 + 9*x, x))  # {3: 2, 0: 1}
# 微分方程 Eq(f(x), (C1 + C2*x)*exp(x) + cos(x)/2)
print("68\t", dsolve(Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), sin(x)), f(x)))
# 微分方程 Eq(f(x) + cos(f(x)), C1)
print("69\t", dsolve(f(x).diff(x)*(1 - sin(f(x))), f(x)))
