import matplotlib.pyplot as plt
import random
import numpy as np
from deap import creator, base, tools
from scipy.stats import bernoulli

# 确保结果可以复现
random.seed(42)
# 描述问题
# 单目标，最大值问题
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
# 编码继承list类
creator.create('Individual', list, fitness=creator.FitnessMax)

# 二进制个体编码
# 需要26位编码
GENE_LENGTH = 26

toolbox = base.Toolbox()
# 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
toolbox.register('binary', bernoulli.rvs, 0.5)
# 用tools.initRepeat生成长度为GENE_LENGTH的Individual
toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.binary, n=GENE_LENGTH)

# 评价函数


def eval(individual):
     # 解码到10进制
    num = int(''.join([str(_) for _ in individual]), 2)
    # 映射回-30，30区间
    x = -30 + num * 60 / (2**26 - 1)
    return ((np.square(x) + x) * np.cos(2*x) + np.square(x) + x),


toolbox.register('evaluate', eval)

# 生成初始族群
# 族群中的个体数量
N_POP = 100
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=N_POP)

# 评价初始族群
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# 进化迭代
# 迭代代数
N_GEN = 50
# 交叉概率
CXPB = 0.5
# 突变概率
MUTPB = 0.2

# 注册进化过程需要的工具：配种选择、交叉、突变
# 注册Tournsize为2的锦标赛选择
toolbox.register('tourSel', tools.selTournament,
                 tournsize=2)
toolbox.register('crossover', tools.cxUniform)
toolbox.register('mutate', tools.mutFlipBit)

# 用数据记录算法迭代过程
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
logbook = tools.Logbook()

for gen in range(N_GEN):
    # 配种选择
    # 选择N_POP个体
    selectedTour = toolbox.tourSel(pop, N_POP)
    # 复制个体，供交叉变异用
    selectedInd = list(map(toolbox.clone, selectedTour))
    # 对选出的育种族群两两进行交叉，对于被改变个体，删除其适应度值
    for child1, child2 in zip(selectedInd[::2], selectedInd[1::2]):
        if random.random() < CXPB:
            toolbox.crossover(child1, child2, 0.5)
            del child1.fitness.values
            del child2.fitness.values

    # 对选出的育种族群进行变异，对于被改变个体，删除适应度值
    for mutant in selectedInd:
        if random.random() < MUTPB:
            toolbox.mutate(mutant, 0.5)
            del mutant.fitness.values

    # 对于被改变的个体，重新评价其适应度
    invalid_ind = [ind for ind in selectedInd if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 完全重插入
    pop[:] = selectedInd

    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen=gen, **record)

# 输出计算过程
logbook.header = 'gen', "avg", "std", 'min', "max"
print(logbook)


def decode(individual):
    # 解码到10进制
    num = int(''.join([str(_) for _ in individual]), 2)
    # 映射回-30，30区间
    x = -30 + num * 60 / (2**26 - 1)
    return x


# 输出最优解
index = np.argmax([ind.fitness for ind in pop])
x = decode(pop[index])  # 解码
print('当前最优解：' + str(x) + '\t对应的函数值为：' + str(pop[index].fitness))

# 结果可视化

gen = logbook.select('gen')  # 用select方法从logbook中提取迭代次数
fit_maxs = logbook.select('max')
fit_average = logbook.select('avg')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(gen, fit_maxs, 'b-', linewidth=2.0, label='Max Fitness')
ax.plot(gen, fit_average, 'r-', linewidth=2.0, label='Average Fitness')
ax.legend(loc='best')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')

fig.tight_layout()
fig.show()

# https://www.jianshu.com/p/3cbf5df95597
