import random
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms
# 问题定义
# 最小化问题
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

# 个体编码
# 两个变量值的下界
low = [-4.5, -4.5]
# 两个变量值的上界
up = [4.5, 4.5]
# 在下界和上界间用均匀分布生成实数变量


def genInd(low, up):
    return [random.uniform(low[0], up[0]), random.uniform(low[1], up[1])]


toolbox = base.Toolbox()
toolbox.register('genInd', genInd, low, up)
toolbox.register('individual', tools.initIterate,
                 creator.Individual, toolbox.genInd)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# 评价函数


def evaluate(ind):
    f = (1.5-ind[0]+ind[0]*ind[1])*(1.5-ind[0]+ind[0]*ind[1]) + \
        (2.25-ind[0]+ind[0]*ind[1]*ind[1])*(2.25-ind[0]+ind[0]*ind[1]*ind[1]) + \
        (2.625-ind[0]+ind[0]*ind[1]*ind[1]*ind[1]) * \
        (2.625-ind[0]+ind[0]*ind[1]*ind[1]*ind[1])
    return f,


toolbox.register('eval', evaluate)

# 记录迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)
stats.register('std', np.std)
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields)

# 注册遗传算法操作 - 选择,交叉,突变
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxSimulatedBinaryBounded,
                 eta=20, low=low, up=up)
toolbox.register('mutate', tools.mutPolynomialBounded,
                 eta=20, low=low, up=up, indpb=0.2)

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
print('最优解为: '+str(bestInd))
print('函数最小值为: '+str(bestFit))

# 结果为
## 最优解为: [3.0283366488777057, 0.5077404970344487]
## 函数最小值为: 0.00013976406990386456

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
