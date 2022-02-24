import random
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms

# 问题定义
# 最小化问题
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

# 个体编码 -- 二进制编码
# 基因长度
geneLen = 48
toolbox = base.Toolbox()
# 二进制编码
toolbox.register('Binary', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat,
                 creator.Individual, toolbox.Binary, geneLen)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# 二进制编码的解码
# 两个变量值的下界
low = [-4.5, -4.5]
# 两个变量值的上界
up = [4.5, 4.5]


def decode(ind, low=low, up=up, geneLen=geneLen):
    '''给定一条二进制编码,将其分割为两个变量,并且转换为对应的十进制数'''
    # 将一条染色体分为两个变量,转换为10进制
    subGeneLen = int(geneLen/2)
    x1 = int(''.join(str(_) for _ in ind[:subGeneLen]), 2)
    x2 = int(''.join(str(_) for _ in ind[subGeneLen:]), 2)
    x1Rescaled = low[0] + x1 * (up[0] - low[0]) / (2**subGeneLen - 1)
    x2Rescaled = low[1] + x2 * (up[1] - low[1]) / (2**subGeneLen - 1)
    return x1Rescaled, x2Rescaled

# 评价函数


def evaluate(ind):
    # 先解码为10进制
    x1, x2 = decode(ind)
    f = (1.5-x1+x1*x2)*(1.5-x1+x1*x2) + \
        (2.25-x1+x1*x2*x2)*(2.25-x1+x1*x2*x2) + \
        (2.625-x1+x1*x2*x2*x2)*(2.625-x1+x1*x2*x2*x2)
    return f,


toolbox.register('eval', evaluate)

# 迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)
stats.register('std', np.std)
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields)

# 注册遗传算法操作 - 选择,交叉,突变
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.5)

# ----------------------
# 遗传算法主要部分
# 族群规模
popSize = 100
# 迭代代数
ngen = 200
# 交叉概率
cxpb = 0.8
# 突变概率
mutpb = 0.1
# 生成初始族群
pop = toolbox.population(popSize)

# 评价初始族群
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.eval, invalid_ind)
for ind, fitness in zip(invalid_ind, fitnesses):
    ind.fitness.values = fitness
record = stats.compile(pop)
logbook.record(gen=0, nevals=len(invalid_ind), **record)

# 遗传算法迭代
for gen in range(1, ngen+1):
    # 育种选择
    # 子代规模与父代相同
    offspring = toolbox.select(pop, popSize)
    offspring = [toolbox.clone(_) for _ in offspring]

    # 变异操作
    # 突变
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
    for ind in offspring:
        if random.random() < mutpb:
            toolbox.mutate(ind)
            del ind.fitness.values

    # 评价子代
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.eval, invalid_ind)
    for ind, fitness in zip(invalid_ind, fitnesses):
        ind.fitness.values = fitness

    # 环境选择
    # 采用精英策略,加速收敛
    combinedPop = pop + offspring
    pop = tools.selBest(combinedPop, popSize)

    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
print(logbook)

# 输出结果
bestInd = tools.selBest(pop, 1)[0]
bestFit = bestInd.fitness.values[0]
print('最优解为: '+str(decode(bestInd)))
print('函数最小值为: '+str(bestFit))

# 结果
## 最优解为: (2.8837195267510136, 0.46988040029289735)
## 函数最小值为: 0.002469514927004176

# 可视化迭代过程
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')

# 保存计算结果
plt.tight_layout()
plt.show()
# https://www.jianshu.com/p/f05f90765496
