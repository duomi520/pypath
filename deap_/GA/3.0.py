import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import random

params = {
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)

# -------------------
## 问题定义
# 最大问题
creator.create('FitnessMax', base.Fitness, weights=(1.0,)) 
creator.create('Individual', list, fitness=creator.FitnessMax)

## 个体编码 - 二进制编码
geneLength = 20
toolbox = base.Toolbox()
toolbox.register('genBinary', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual,toolbox.genBinary, geneLength)

## 评价函数
weightList = [2, 5, 18, 3, 2, 5, 10, 4, 8, 12, 5, 10, 7, 15, 11, 2, 8, 10, 5, 9]
valueList = [5, 10, 12, 4, 3, 11, 13, 10, 7, 15, 8, 19, 1, 17, 12, 9, 15, 20, 2, 6]
def evaluate(ind):
    return (np.sum(np.asarray(valueList)*np.asarray(ind))),
toolbox.register('evaluate', evaluate)

## 施加惩罚
def feasible(ind, W=40):
    '''可行性函数，判断个体是否满足背包总重量约束'''
    weightSum = np.sum(np.asarray(weightList)*np.asarray(ind))
    if weightSum <= W:
        return True
    return False
# 死亡惩罚，当个体不满足总重量约束时，设置其个体适应度为-10
toolbox.decorate('evaluate', tools.DeltaPenalty(feasible, -10))

## 注册工具
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.5)

# 生成族群
popSize = 100
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(popSize)

# 迭代数据
stats = tools.Statistics(key=lambda ind:ind.fitness.values)
stats.register('max', np.max)
stats.register('avg', np.mean)
stats.register('std', np.std)
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields)

# -------------------
# 评价初始族群
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for fitness, ind in zip(fitnesses, invalid_ind):
    ind.fitness.values = fitness

# 记录第0代的数据
record = stats.compile(pop)
logbook.record(gen=0, nevals=len(invalid_ind),**record)

# 参数设置
ngen = 200
cxpb = 0.8
mutpb = 0.2

# 遗传算法迭代
for gen in range(1, ngen+1):
    # 育种选择
    offspring = toolbox.select(pop, popSize)
    offspring = [toolbox.clone(_) for _ in offspring]
    # 交叉
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random()<cxpb:
            toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
    # 突变
    for ind in offspring:
        if random.random()<mutpb:
            toolbox.mutate(ind)
            del ind.fitness.values
    # 评价子代
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for fitness, ind in zip(fitnesses, invalid_ind):
        ind.fitness.values = fitness
    # 环境选择
    combindPop = pop + offspring
    pop = tools.selBest(combindPop, popSize)
    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen = gen, nevals = len(invalid_ind), **record)
print(logbook)

## 输出结果
bestInd = tools.selBest(pop,1)[0]
bestFit = bestInd.fitness.values[0]
weightSum = np.sum(np.asarray(weightList)*np.asarray(bestInd))
print('最优解为: '+str(bestInd))
print('函数最大值为: '+str(bestFit))
print('背包重量为: '+str(weightSum))

## 可视化迭代过程
maxFit = logbook.select('max')
avgFit = logbook.select('avg')
plt.plot(maxFit, 'b-', label='Maximum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')

## 结果
# 最优解为: [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
# 函数最大值为: 83.0
# 背包重量为: 39

# https://www.jianshu.com/p/0f61bffe6a1e