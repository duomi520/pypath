import random
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms
from scipy.spatial import distance


def genCity(n, Lb=100, Ub=999):
    # 生成城市坐标
    # 输入：n -- 需要生成的城市数量
    # 输出: nx2 np array 每行是一个城市的[X,Y]坐标
    # 保证结果的可复现性
    np.random.seed(42)
    return np.random.randint(low=Lb, high=Ub, size=(n, 2))
# 计算并存储城市距离矩阵


def cityDistance(cities):
    # 生成城市距离矩阵 distMat[A,B] = distMat[B,A]表示城市A，B之间距离
    # 输入：cities -- [n,2] np array， 表示城市坐标
    # 输出：nxn np array， 存储城市两两之间的距离
    return distance.cdist(cities, cities, 'euclidean')


def completeRoute(individual):
    # 序列编码时，缺少最后一段回到原点的线段
    # 不要用append
    return individual + [individual[0]]

# 计算给定路线的长度


def routeDistance(route):
    # 输入：
    #      route -- 一条路线，一个sequence
    # 输出：routeDist -- scalar，路线的长度
    if route[0] != route[-1]:
        route = completeRoute(route)
    routeDist = 0
    # 这里直接从cityDist变量中取值了，其实并不是很安全的写法，单纯偷懒了
    for i, j in zip(route[0::], route[1::]):
        routeDist += cityDist[i, j]
    # 注意DEAP要求评价函数返回一个元组
    return (routeDist),


def GA_improved(cities, nCities, npop=100, cxpb=0.5, mutpb=0.2, ngen=200):
    global cityDist
    # 问题定义
    # 最小化问题
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    # 定义个体编码
    toolbox = base.Toolbox()
    # 创建序列
    toolbox.register('indices', random.sample, range(nCities), nCities)
    toolbox.register('individual', tools.initIterate,
                     creator.Individual, toolbox.indices)

    # 生成族群
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(npop)

    # 注册所需工具
    cityDist = cityDistance(cities)
    toolbox.register('evaluate', routeDistance)
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.2)

    # 数据记录
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('min', np.min)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields)

    # 实现遗传算法
    # 评价族群
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    # 方便输出数据好看
    for gen in range(1, ngen+1):
        # 配种选择
        offspring = toolbox.select(pop, 2*npop)
        # 一定要复制，否则在交叉和突变这样的原位操作中，会改变所有select出来的同个体副本
        offspring = [toolbox.clone(_) for _ in offspring]
        # 变异操作 - 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # 变异操作 - 突变
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # 评价当前没有fitness的个体，确保整个族群中的个体都有对应的适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # 环境选择 - 保留精英
        # 选择精英,保持种群规模
        pop = tools.selBest(offspring, npop, fit_attr='fitness')
#         pop[:] = offspring
        # 记录数据
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    return pop, logbook


nCities = 30
# 随机生成nCities个城市坐标
cities = genCity(nCities)
resultPopGA_improved, logbookGA_improved = GA_improved(cities, nCities)
plt.plot(logbookGA_improved.select('min'), 'r-', label='Minimum Fitness')
plt.plot(logbookGA_improved.select('avg'), 'b-', label='Average Fitness')
plt.ylabel('Fitness')
plt.xlabel('# Iteration')
plt.legend(loc='best')
plt.tight_layout()
plt.title(
    f'GA with eliteness preservation strategy iterations, Problem size:{nCities}')
plt.show()

def plotTour(tour, cities, style='bo-'):
    if len(tour) > 1000:
        plt.figure(figsize=(15, 10))
    start = tour[0:1]
    for i, j in zip(tour[0::], tour[1::]):
        plt.plot([cities[i, 0], cities[j, 0]], [
                 cities[i, 1], cities[j, 1]], style)
    plt.plot(cities[start, 0], cities[start, 1], 'rD')
    plt.axis('scaled')
    plt.axis('off')
    plt.show()


# 对结果进行可视化
tour = tools.selBest(resultPopGA_improved, k=1)[0]
tourDist = tour.fitness
tour = completeRoute(tour)
print('最优路径为:'+str(tour))
print('最优路径距离为：'+str(tourDist))
plotTour(tour, cities)

# https://www.jianshu.com/p/a15d06645767