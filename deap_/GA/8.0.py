from pprint import pprint
from copy import deepcopy
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

# 问题描述
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

# 个体编码
sourceFlow = [8, 4, 12, 6]
destinationFlow = [3, 5, 10, 7, 5]


def matEncode(sourceFlow=sourceFlow, destinationFlow=destinationFlow):
    '''生成可行的配送量矩阵'''
    rowNum = len(sourceFlow)
    colNum = len(destinationFlow)
    # 当前未填充的位置集合
    posSet = list(range(rowNum * colNum))
    # 初始化染色体矩阵
    mat = np.zeros((rowNum, colNum))
    sourceFlowCopy = deepcopy(sourceFlow)
    destinationFlowCopy = deepcopy(destinationFlow)
    while posSet:
        # 随机选取位置集中的一个元素
        randIdx = random.randint(0, len(posSet)-1)
        posNum = posSet[randIdx]
        # 第rowPos行
        rowPos = (posNum - 1)//colNum + 1
        # 第colPos列
        colPos = posNum % colNum
        # 填充矩阵相应位置
        mat[rowPos-1, colPos -
            1] = min(sourceFlowCopy[rowPos-1], destinationFlowCopy[colPos-1])
        # 根据填充值，修改剩余的产量和销量
        sourceFlowCopy[rowPos-1] -= mat[rowPos-1, colPos-1]
        destinationFlowCopy[colPos-1] -= mat[rowPos-1, colPos-1]
        # 从位置集合中删去已经被填充的位置
        posSet = posSet[:randIdx] + posSet[randIdx+1:]
    return mat.tolist()


# Cost Matrix
costMat = [
    [8.6860922, 9.35319846, 4.04839742, 6.64822502, 2.21060537],
    [2.09958268, 7.93166898, 5.69745574, 4.18627465, 6.11192203],
    [7.05011006, 9.65028246, 5.79551805, 2.74632325, 5.96366128],
    [6.66228395, 8.73934611, 4.89579209, 6.13292158, 5.65538837]
]

# 基于优先级的编码


def priorityCoding(matCode, costMat=costMat, sourceFlow=sourceFlow, destinationFlow=destinationFlow):
    '''
    从给定的配送量矩阵和代价矩阵,生成基于优先级的编码
    输入: matCode -- 一个满足sourceFlow和destinationFlow的可行解
    costMat -- 代价矩阵,给出由一个source到一个destination的代价
    sourceFlow, destinationFlow -- 每个source的输出能力和destination的接受能力
    输出: 长度为len(sourceFlow) + len(destinationFlow)的编码
    '''
    # 初始化编码
    priorityCode = [0] * (len(sourceFlow) + len(destinationFlow))
    # 初始化优先级
    priority = len(priorityCode)
    # 复制矩阵,防止改变原值
    sourceFlowCopy = deepcopy(sourceFlow)
    destinationFlowCopy = deepcopy(destinationFlow)
    matCodeCopy = deepcopy(matCode)
    costMatCopy = deepcopy(costMat)
    largeNum = 1e5  # 设定一个足够大的数
    # 当流量没有完全分配时,执行迭代
    while not (np.any(sourceFlowCopy) and np.any(destinationFlowCopy)):
        # 为配送量为0的连接分配较大代价
        costMatCopy = np.where(matCodeCopy == 0, largeNum, costMatCopy)
        # 最小运输代价所对应的source和destination
        i, j = np.where(costMatCopy=np.min(costMatCopy))
        # 更新剩余source和destination流量
        sourceFlowCopy[i] -= matCodeCopy[i][j]
        destinationFlowCopy[j] -= matCodeCopy[i][j]
        # 当剩余sourceFlow或者destinationFlow为0时,分配优先度编码
        if sourceFlowCopy[i] == 0:
            priorityCode[i] = priority
            priority -= 1
        if destinationFlowCopy[j] == 0:
            priorityCode[j + len(sourceFlowCopy)] = priority
            priority -= 1
    # 为剩余位置填充优先级,因为该边上实质流量为0,因此事实上非有效边,可以随意分配流量
    perm = np.random.permutation(range(1, priority + 1))
    priorityCode = np.asarray(priorityCode)
    priorityCode[priorityCode == 0] = perm
    return priorityCode


# 解码
def priorityDecoding(priorityCode, costMat=costMat, sourceFlow=sourceFlow, destinationFlow=destinationFlow):
    '''根据优先度编码解码回配送方案的矩阵编码
    输入:priorityCode -- 基于优先级的编码, 长度为len(sourceFlow) + len(destinationFlow)
    costMat -- 代价矩阵,给出由一个source到一个destination的代价
    sourceFlow, destinationFlow -- 每个source的输出能力和destination的接受能力
    输出:matCode -- 一个满足sourceFlow和destinationFlow的可行解矩阵编码'''
    # 初始化矩阵编码
    matCode = np.zeros((len(sourceFlow), len(destinationFlow)))
    # 复制矩阵,防止改变原值
    sourceFlowCopy = deepcopy(sourceFlow)
    destinationFlowCopy = deepcopy(destinationFlow)
    costMatCopy = np.array(deepcopy(costMat))
    priorityCodeCopy = deepcopy(priorityCode)
    # 设定一个足够大的数
    largeNum = 1e5
    # 列出source Node和destination Node
    sourceNodeList = list(range(1, len(sourceFlow)+1))
    destinationNodeList = list(range(len(sourceFlow)+1,
                                     len(destinationFlow)+len(sourceFlow)+1))
    nodeList = sourceNodeList + destinationNodeList
    while np.any(priorityCodeCopy):
        # 选择优先度最高的节点
        nodeSelected = np.asarray(nodeList)[np.argmax(priorityCodeCopy)]
        # 为剩余流量为0的行和列分配一个大数作为运输代价
        rowIdx = [i for i in range(len(sourceFlowCopy))
                  if sourceFlowCopy[i] == 0]
        colIdx = [i for i in range(
            len(destinationFlowCopy)) if destinationFlowCopy[i] == 0]
        for row in rowIdx:
            costMatCopy[row, :] = [largeNum]*len(costMatCopy[row, :])
        for col in colIdx:
            costMatCopy[:, col] = [largeNum]*len(costMatCopy[:, col])
        # 如果选中的节点在供方,则从需方选择对应运输代价最小的节点
        if nodeSelected in sourceNodeList:
            # 作为index,比节点标号小1
            sourceNodeIdx = nodeSelected - 1
            destinationNodeIdx = np.argmin(costMatCopy[sourceNodeIdx, :])
        # 如果选中的节点在需方,则从供方选择对应运输代价最小的节点
        else:
            destinationNodeIdx = nodeSelected - 1 - len(sourceFlow)
            sourceNodeIdx = np.argmin(costMatCopy[:, destinationNodeIdx])
        # 更新选中边上的流量
        matCode[sourceNodeIdx][destinationNodeIdx] = min(sourceFlowCopy[sourceNodeIdx],
                                                         destinationFlowCopy[destinationNodeIdx])
        # 更新剩余流量
        sourceFlowCopy[sourceNodeIdx] -= matCode[sourceNodeIdx][destinationNodeIdx]
        destinationFlowCopy[destinationNodeIdx] -= matCode[sourceNodeIdx][destinationNodeIdx]
        # 更新优先度
        if sourceFlowCopy[sourceNodeIdx] == 0:
            priorityCodeCopy[sourceNodeIdx] = 0
        if destinationFlowCopy[destinationNodeIdx] == 0:
            priorityCodeCopy[destinationNodeIdx + len(sourceFlowCopy)] = 0
    return matCode

# 评价函数


def evaluate(ind):
    # 将个体优先度编码解码为矩阵编码
    matCode = priorityDecoding(ind)
    return np.sum(np.sum(np.asarray(costMat)*np.asarray(matCode))),

# DEAP原生的PMX需要个体编码从0开始


def PMX(ind1, ind2):
    ind1Aligned = [ind1[i]-1 for i in range(len(ind1))]
    ind2Aligned = [ind2[i]-1 for i in range(len(ind2))]
    tools.cxPartialyMatched(ind1Aligned, ind2Aligned)
    ind1Recovered = [ind1Aligned[i]+1 for i in range(len(ind1Aligned))]
    ind2Recovered = [ind2Aligned[i]+1 for i in range(len(ind2Aligned))]
    ind1[:] = ind1Recovered
    ind2[:] = ind2Recovered
    return ind1, ind2


# 注册工具
toolbox = base.Toolbox()
toolbox.register('genPriority', priorityCoding, matEncode())
toolbox.register('individual', tools.initIterate,
                 creator.Individual, toolbox.genPriority)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', PMX)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.5)

# 生成初始族群
toolbox.popSize = 100
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(toolbox.popSize)

# 记录迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)
hallOfFame = tools.HallOfFame(maxsize=1)

# 遗传算法参数
toolbox.ngen = 500
toolbox.cxpb = 0.8
toolbox.mutpb = 0.1


# ## 遗传算法主程序
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
                                   ngen=toolbox.ngen, stats=stats, halloffame=hallOfFame, verbose=True)

# 输出结果
bestInd = hallOfFame.items[0]
bestFitness = bestInd.fitness.values
print('最佳运输组合为：')
pprint(priorityDecoding(bestInd))
print('该运输组合的代价为：'+str(bestFitness))

# 画出迭代图
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()
# 计算结果：
# 最佳运输组合为：
# array([[0., 0., 3., 0., 5.],
#       [3., 1., 0., 0., 0.],
#       [0., 0., 5., 7., 0.],
#       [0., 4., 2., 0., 0.]])
# 该运输组合的代价为：(130.37945775,)

# https://www.jianshu.com/p/20c653bb20fe