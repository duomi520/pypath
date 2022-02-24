import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------PSO参数设置---------------------------------


class PSO():
    def __init__(self, func, pN, dim, max_iter):
        self.func = func
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值

    # ---------------------目标函数Sphere函数-----------------------------
    def function(self, X):
        return self.func(X)

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 1)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i]

    # ----------------------更新粒子位置----------------------------------

    def iterator(self):
        fitness = []
        for _ in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                    self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
        return fitness
    # ----------------------程序执行-----------------------


if __name__ == '__main__':
    my_pso = PSO(lambda x: x**2-4*x+3, pN=30, dim=1, max_iter=100)
    my_pso.init_Population()
    fitness = my_pso.iterator()
    print('最优目标函数值:', min(fitness))
    # -------------------画图--------------------
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 100)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='b', linewidth=3)
    plt.show()


# http://www.omegaxyz.com/2018/01/12/python_pso/
