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

from copy import deepcopy
# 问题定义
# 最小化问题
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

# 个体编码
edges = [
    '1,2', '1,3', '1,4', '1,5', '1,6',
    '2,3', '2,4', '2,5', '2,6',
    '3,4', '3,7', '3,8', '3,9', '3,10',
    '4,5', '4,7', '4,8', '4,9', '4,10',
    '5,6', '5,7', '5,8', '5,9', '5,10',
    '6,7', '6,8', '6,9', '6,10',
    '7,8', '7,11', '7,12',
    '8,9', '8,11', '8,12',
    '9,10', '9,11', '9,12',
    '10,11', '10,12',
    '11,12'
]
def generateSFromEdges(edges):
    '''用关联表存储图，从提供的边集中生成与各个节点i相邻的节点集合Si
    输入：edges -- list, 其中每个元素为每个节点上的边，每个元素均为一个str 'i,j'
    输出：nodeDict -- dict, 形如{'i':[j,k,l]}，记录从每个节点能到达的其他节点
    '''
    nodeDict = {}
    for edge in edges:
        i,j = edge.split(',')
        if not i in nodeDict:
            nodeDict[i] = [int(j)]
        else:
            nodeDict[i].append(int(j))
        # 无向图中(i,j)与(j,i)是相同的
        if not j in nodeDict:
            nodeDict[j] = [int(i)]
        else:
            nodeDict[j].append(int(i))
    return nodeDict

def eligibleEdgeSet(nodeDict, i):
    '''辅助函数，生成从节点i出发的所有边(i,j)
    输入：nodeDict -- dict，记录每个节点能到达的其他节点
    i -- 起始节点，int
    输出：edgeSet -- list，记录从节点i出发可能的所有边的集合，其中每个元素为一条边，形如'i,j'的str
    '''
    # i节点的所有后续节点
    endNodeSet = nodeDict[str(i)] 
    edgeSet = []
    for eachNode in endNodeSet:
        edgeSet.append(str(i) + ',' + str(eachNode))
    return edgeSet
    
def genEdgeFromNodeSet(nodeSet):
    '''辅助函数，给定一个noseSet，返回其中所有可能的边
    输入： nodeSet -- list，每个元素均为int,代表一个节点
    输出：edgesGen -- list, 每个元素代表一条边，形如'i,j'的str
    '''
    from itertools import combinations
    combs = combinations(nodeSet, 2)
    edgesGen = []
    for eachItem in combs:
        edgesGen.append(str(eachItem[0])+','+str(eachItem[1]))
        edgesGen.append(str(eachItem[1])+','+str(eachItem[0]))
    return edgesGen
    
def PrimPredCoding(edges=edges):
    '''从给定的节点集合中以PrimPred方法生成染色体
    输入：
    输出：ind -- 个体实数编码，长度为节点数-1
    '''
    nodeDict = generateSFromEdges(edges)
    # 这个长度等于节点数
    nodeCount = len(nodeDict) 
    i = 1
    # 用于保存迭代中间变量
    nodeSet = [i] 
    # 从i出发所有可能的边
    edgeSet = eligibleEdgeSet(nodeDict, i) 
    iterIdx = 1
    # [1]作为默认起始点，需要的编码长度为节点数-1
    ind = [0]*(nodeCount-1) 
    while iterIdx < nodeCount:
        # 随机选取一条可行边
        edgeSelected = edgeSet[random.randint(0,len(edgeSet)-1)] 
         # 所选边的起点
        i = int(edgeSelected.split(',')[0])
         # 所选边的终点，范围为2到len(nodeDict)+1
        j = int(edgeSelected.split(',')[1])
#         print(len(ind), j-1)
# 注意j是从1到len(#node)的，作为index应该减去1
        ind[j-2] = i 
        nodeSet.append(j)
        i = j
         # 当i时最终节点时，没有可用的边了
        if not i==len(nodeDict) +1:
            edgeSet = edgeSet + eligibleEdgeSet(nodeDict, i)
        # 需要从集合中删掉的边
        edgesToExclude = genEdgeFromNodeSet(nodeSet) 
        edgeSet = list(set(edgeSet) - set(edgesToExclude))
        iterIdx += 1
    return ind

toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual ,PrimPredCoding)

# 解码
def decoding(ind):
    '''对给定的染色体编码，解码为生成树(边的集合)
    输入：ind -- 个体实数编码，长度为节点数-1
    输出：generatedTree -- 边的集合，类似于edges，每个元素为形如'i,j'的str
    '''
    generatedTree = []
    geneLen = len(ind)
    for i,j in zip(ind, range(2,2+geneLen)):
        generatedTree.append(str(min(i,j))+','+str(max(i,j)))
    return generatedTree

# 评价个体
weightDict = {
    '1,2': 35, '1,3': 23, '1,4': 26, '1,5': 29, '1,6': 52,
    '2,3': 34, '2,4': 23, '2,5': 68, '2,6': 42,
    '3,4': 23, '3,7': 51, '3,8': 23, '3,9': 64, '3,10': 28,
    '4,5': 54, '4,7': 24, '4,8': 47, '4,9': 53, '4,10': 24,
    '5,6': 56, '5,7': 26, '5,8': 35, '5,9': 63, '5,10': 23,
    '6,7': 27, '6,8': 29, '6,9': 65, '6,10': 24,
    '7,8': 38, '7,11': 52, '7,12': 41,
    '8,9': 62, '8,11': 26, '8,12': 30,
    '9,10': 47, '9,11': 68, '9,12': 33,
    '10,11': 42, '10,12': 26,
    '11,12': 51
}
def evaluate(ind):
    '''对给定的染色体编码，返回给定边的权值之和'''
    generatedTree = decoding(ind)
    weightSum = 0
    for eachEdge in generatedTree:
        weightSum += weightDict[eachEdge]
    return weightSum,

# 交叉操作
def cxPrimPred(ind1, ind2):
    '''给定两个个体，将其边叠加，再根据PrimPred编码方法生成新个体'''
    # 将个体解码为边
    edges1 = decoding(ind1) 
    edges2 = decoding(ind2)
    edgesCombined = list(set(edges1 + edges2))
    return PrimPredCoding(edges=edgesCombined)

# 突变操作
def mutLowestCost(ind, weightDict=weightDict):
    '''给定一个个体，用lowest cost method生成新个体，先将父代染色体中随机删除一条边，将原图
    分为两个互不连通的子图，然后选择连接这两个子图的具有最小权数的边并连接子图'''
    # 将原图分为两个互不连通的子图
    edges = decoding(ind)
     # 选择一条需要删除的边
    edgeIdx = random.randint(0, len(edges)-1)
    u = int(edges[edgeIdx].split(',')[0])
    v = int(edges[edgeIdx].split(',')[1])
    # 删除选中的边
    edges = edges[:edgeIdx] + edges[edgeIdx+1:] 
    # 将属于两个子图的顶点分别归如两个点集
    A = [0]*(len(ind)+1)
    U = edges
    while U:
        # 随机选择一条边(i,j)
        randomEdgeIdx = random.randint(0, len(U)-1) 
        i = int(U[randomEdgeIdx].split(',')[0])
        j = int(U[randomEdgeIdx].split(',')[1])
        # 删除选中的边
        U = U[:randomEdgeIdx] + U[randomEdgeIdx+1:] 
        if A[i-1] == 0 and A[j-1] == 0:
            l = min(i,j)
            A[i-1] = l
            A[j-1] = l
        elif A[i-1] == 0 and A[j-1] != 0:
            A[i-1] = A[j-1]
        elif A[i-1] != 0 and A[j-1] == 0:
            A[j-1] = A[i-1]
        else:
            if A[i-1] < A[j-1]:
                idx = [A[_]==A[j-1] for _ in range(len(A))]
                A = np.where(idx, A[i-1], A)
            elif A[i-1] > A[j-1]:
                idx = [A[_]==A[i-1] for _ in range(len(A))]
                A = np.where(idx, A[j-1], A)
    # 注意index和节点编号的关系
    nodeSet1 = [_+1 for _ in range(len(A)) if A[_]==A[u-1]] 
    nodeSet2 = [_+1 for _ in range(len(A)) if A[_]==A[v-1]]
    # 选择两个点集中代价最小的边，添进边中
    minCostEdge = None
    minEdgeCost = 1e5
    for vert1 in nodeSet1:
        for vert2 in nodeSet2:
            key = str(min(vert1,vert2)) + ',' + str(max(vert1,vert2))
            if key in weightDict:
                if weightDict[key] < minEdgeCost:
                    minEdgeCost = weightDict[key]
                    minCostEdge = key
    edges = edges + [minCostEdge]
    # 从边还原编码
    return PrimPredCoding(edges)

# 注册工具
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', cxPrimPred)
toolbox.register('mutate', mutLowestCost)

# 迭代数据
stats = tools.Statistics(key=lambda ind:ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)

##---------------------------
# 遗传算法参数
toolbox.ngen = 200
toolbox.popSize = 100
toolbox.cxpb = 0.8
toolbox.mutpb = 0.1

# 生成初始族群
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(toolbox.popSize)

# 遗传算法主程序
hallOfFame = tools.HallOfFame(maxsize=1)
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit

hallOfFame.update(pop)

record = stats.compile(pop) if stats else {}
logbook.record(gen=0, nevals=len(invalid_ind), **record)

# Begin the generational process
for gen in range(1, toolbox.ngen + 1):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Vary the pool of individuals
    for i in range(1, len(offspring), 2):
        if random.random() < toolbox.cxpb:
            offspring[i - 1][:] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values

    for i in range(len(offspring)):
        if random.random() < toolbox.mutpb:
            offspring[i][:] = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the generated individuals
    hallOfFame.update(offspring)

    # Replace the current population by the offspring
    pop[:] = offspring

    # Append the current generation statistics to the logbook
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
print(logbook)

# 输出结果
bestInd = hallOfFame.items[0]
bestFitness = bestInd.fitness.values
bestEdges = decoding(bestInd)
print('最小生成树的边为：'+str(bestEdges))
print('最小生成树的代价为：'+str(bestFitness))

## 画出迭代图
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()

## 结果：
#最小生成树的边为：['2,4', '1,3', '3,4', '5,10', '6,10', '4,7', '3,8', '9,12', '4,10', '8,11', '10,12']
#最小生成树的代价为：(272.0,)

# https://www.jianshu.com/p/2312803c2a82