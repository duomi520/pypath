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
# ----------------------
# 问题定义
# 最大化问题
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

# 个体编码
geneLength = 13
toolbox = base.Toolbox()
toolbox.register('genASCII', random.randint, 97, 122)
toolbox.register('individual', tools.initRepeat,
                 creator.Individual, toolbox.genASCII, n=geneLength)

# 评价函数 -- 生成的字符串与目标字符串相同字符个数


def evaluate(ind):
    target = list('tobeornottobe')
    target = [ord(item) for item in target]
    return (sum(np.asarray(target) == np.asarray(ind))),


toolbox.register('evaluate', evaluate)

# 迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('max', np.max)
stats.register('min', np.min)
stats.register('std', np.std)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields)

# 生成族群
popSize = 100
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(popSize)

# 注册遗传算法操作 -- 选择,交叉,突变
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxUniform, indpb=0.5)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.5)

# ----------------------
# 遗传算法
# 迭代步数
ngen = 400
# 交叉概率
cxpb = 0.8
# 突变概率
mutpb = 0.2
# 评价初始族群
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, pop)
for fitness, ind in zip(fitnesses, invalid_ind):
    ind.fitness.values = fitness

# 记录数据
record = stats.compile(pop)
logbook.record(gen=0, nevals=len(invalid_ind), **record)

# 迭代
for gen in range(1, ngen+1):
    # 配种选择
    offspring = toolbox.select(pop, popSize)
    offspring = [toolbox.clone(_) for _ in offspring]
    # 交叉
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
    # 突变
    for ind in offspring:
        if random.random() < mutpb:
            toolbox.mutate(ind)
            del ind.fitness.values
    # 评价子代
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for fitness, ind in zip(fitnesses, invalid_ind):
        ind.fitness.values = fitness
    # 环境选择
    combinedPop = pop + offspring
    # 精英保存策略
    pop = tools.selBest(combinedPop, popSize)
    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
print(logbook)
# 输出结果
bestInd = tools.selBest(pop, 1)[0]
bestFit = bestInd.fitness.values[0]
bestInd = [chr(item) for item in bestInd]
print('最优解为: '+str(bestInd))
print('函数最大值为: '+str(bestFit))

# 结果
## 最优解为: ['t', 'o', 'b', 'e', 'o', 'r', 'n', 'o', 't', 't', 'o', 'b', 'e']
## 函数最大值为: 13.0
# 可视化迭代过程
maxFit = logbook.select('max')
avgFit = logbook.select('avg')
plt.plot(maxFit, 'b-', label='Maximum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()

# https://www.jianshu.com/p/ff153901ef8e
