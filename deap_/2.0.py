import random
import numpy as np
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli

# 确保可以复现结果
random.seed(42)
# 描述问题
# 单目标，最大值问题
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
# 编码继承list类
creator.create('Individual', list, fitness=creator.FitnessMax)

# 个体编码
# 需要26位编码
GENE_LENGTH = 26

toolbox = base.Toolbox()
# 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
toolbox.register('binary', bernoulli.rvs, 0.5)
# 用tools.initRepeat生成长度为GENE_LENGTH的Individual
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.binary, n=GENE_LENGTH)

# 评价函数


def decode(individual):
    # 解码到10进制
    num = int(''.join([str(_) for _ in individual]), 2)
    # 映射回-30，30区间
    x = -30 + num * 60 / (2**26 - 1)
    return x


def eval(individual):
    x = decode(individual)
    return ((np.square(x) + x) * np.cos(2*x) + np.square(x) + x),


# 生成初始族群
# 族群中的个体数量
N_POP = 100
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=N_POP)

# 在工具箱中注册遗传算法需要的工具
toolbox.register('evaluate', eval)
# 注册Tournsize为2的锦标赛选择
toolbox.register('select', tools.selTournament,
                 tournsize=2)
# 注意这里的indpb需要显示给出
toolbox.register('mate', tools.cxUniform, indpb=0.5)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.5)

# 注册计算过程中需要记录的数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# 调用DEAP内置的算法
resultPop, logbook = algorithms.eaSimple(
    pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, verbose=False)

# 输出计算过程
logbook.header = 'gen', 'nevals', "avg", "std", 'min', "max"
print(logbook)

# 输出最优解
index = np.argmax([ind.fitness for ind in resultPop])
x = decode(resultPop[index])  # 解码
print('当前最优解：' + str(x) + '\t对应的函数值为：' + str(resultPop[index].fitness))

# https://www.jianshu.com/p/3cbf5df95597
