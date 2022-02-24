import numpy as np
from scipy.optimize import minimize

# Rosenbrock's "Banana" 测试函数


def banana(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2


x0 = np.array([-1.2, 1])
res = minimize(banana, x0)
print(res)
