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

import copy
# --------------------
# 问题定义
# 最小化问题
creator.create('FitnessMax', base.Fitness, weights=(-1.0,))  
creator.create('Individual', list, fitness=creator.FitnessMax)

# 个体编码
toolbox = base.Toolbox()
geneLength = 25
toolbox.register('perm', np.random.permutation, geneLength)
toolbox.register('individual', tools.initIterate,
                 creator.Individual, toolbox.perm)

# 评价函数
# 类似链表，存储每个节点的可行路径，用于解码
nodeDict = {'1': [2,3,4,5,6], '2': [7,8], '3': [7,8,9], '4': [8,9,10], '5': [9,10,11], 
            '6': [10,11], '7': [13], '8': [12,14], '9': [13,15], '10': [14,16],
            '11':[15], '12':[17], '13':[17,18], '14':[18,19], '15':[19,20],
            '16':[20,21], '17':[18,22], '18':[19,22,23], '19':[20,23,24],
            '20':[23,24], '21':[24], '22':[25], '23':[25], '24':[25]
           }

def genPath(ind, nodeDict=nodeDict):
    '''输入一个优先度序列之后，返回一条从节点1到节点25的可行路径 '''
    path = [1]
    endNode = len(ind)
    while not path[-1] == endNode:
        # 当前所在节点
        curNode = path[-1]  
        # 当前节点指向的下一个节点不为空时，到达下一个节点
         # 获取可以到达的下一个节点列表
        if nodeDict[str(curNode)]:  
            toBeSelected = nodeDict[str(curNode)] 
        else:
            return path
            # 获取优先级,注意列表的index是0-9
        priority = np.asarray(ind)[np.asarray(
            toBeSelected)-1]  
        nextNode = toBeSelected[np.argmax(priority)]
        path.append(nextNode)
    return path


# 存储每条边的剩余容量，用于计算路径流量和更新节点链表
capacityDict = {'1,2': 20, '1,3': 20, '1,4': 20, '1,5': 20, '1,6': 20,
                '2,7': 10, '2,8': 8,
               '3,7': 6, '3,8': 5, '3,9': 4, 
                '4,8': 5, '4,9': 8, '4,10': 10,
                '5,9': 10, '5,10': 4, '5,11': 10,
                '6,10': 10, '6,11': 10,
                '7,13': 15,
                '8,12': 15, '8,14': 15,
                '9,13': 15, '9,15': 15,
                '10,14': 15, '10,16': 15,
                '11,15': 15,
                '12,17': 20,
                '13,17': 20, '13,18': 20,
                '14,18': 20, '14,19': 20,
                '15,19': 20, '15,20': 20,
                '16,20': 20, '16,21': 20,
                '17,18': 8, '17,22': 25,
                '18,19': 8, '18,22': 25, '18,23': 20,
                '19,20': 8, '19,23': 10, '19,24': 25,
                '20,23': 10, '20,24': 25,
                '21,24': 25,
                '22,25': 30, '23,25': 30, '24,25':30}

# 存储每条边的代价
costDict = {'1,2': 10, '1,3': 13, '1,4': 32, '1,5': 135, '1,6': 631,
                '2,7': 10, '2,8': 13,
               '3,7': 10, '3,8': 15, '3,9': 33, 
                '4,8': 4, '4,9': 15, '4,10': 35,
                '5,9': 3, '5,10': 13, '5,11': 33,
                '6,10': 7, '6,11': 7,
                '7,13': 10,
                '8,12': 4, '8,14': 9,
                '9,13': 11, '9,15': 12,
                '10,14': 9, '10,16': 14,
                '11,15': 5,
                '12,17': 8,
                '13,17': 6, '13,18': 7,
                '14,18': 7, '14,19': 7,
                '15,19': 5, '15,20': 14,
                '16,20': 4, '16,21': 14,
                '17,18': 11, '17,22': 11,
                '18,19': 5, '18,22': 8, '18,23': 34,
                '19,20': 3, '19,23': 10, '19,24': 35,
                '20,23': 3, '20,24': 14,
                '21,24': 12,
                '22,25': 10, '23,25': 2, '24,25':3}

def traceCapacity(path, capacityDict):
    ''' 获取给定path的最大流量，更新各边容量 '''
    pathEdge = list(zip(path[::1], path[1::1]))
    keys = []
    edgeCapacity = []
    for edge in pathEdge:
        key = str(edge[0]) + ',' + str(edge[1])
         # 保存edge对应的key
        keys.append(key) 
         # 该边对应的剩余容量
        edgeCapacity.append(capacityDict[key]) 
         # 路径上的最大流量
    pathFlow = min(edgeCapacity) 
    # 更新各边的剩余容量
    for key in keys:
         # 注意这里是原位修改
        capacityDict[key] -= pathFlow 
    return pathFlow

def updateNodeDict(capacityDict, nodeDict):
    ''' 对剩余流量为0的节点，删除节点指向；对于链表指向为空的节点，由于没有下一步可以移动的方向，
    从其他所有节点的指向中删除该节点
    '''
    for edge, capacity in capacityDict.items():
        if capacity == 0:
            # 用来索引节点字典的key，和需要删除的节点toBeDel
            key, toBeDel = str(edge).split(',')  
            if int(toBeDel) in nodeDict[key]:
                nodeDict[key].remove(int(toBeDel))
    delList = []
    for node, nextNode in nodeDict.items():
         # 如果链表指向为空的节点，从其他所有节点的指向中删除该节点
        if not nextNode: 
            delList.append(node)
    for delNode in delList:
        for node, nextNode in nodeDict.items():
            if delNode in nextNode:
                nodeDict[node].remove(delNode)

def calCost(path, pathFlow, costDict):
    '''计算给定路径的成本'''
    pathEdge = list(zip(path[::1], path[1::1]))
    keys = []
    edgeCost = []
    for edge in pathEdge:
        key = str(edge[0]) + ',' + str(edge[1])
        # 保存edge对应的key
        keys.append(key)  
        # 该边对应的cost 
        edgeCost.append(costDict[key])  
    pathCost = sum([eachEdgeCost*pathFlow for eachEdgeCost in edgeCost])
    return pathCost

def evaluate(ind, outputPaths=False):
    '''评价函数'''
    # 初始化所需变量
    # 浅复制
    nodeDictCopy = copy.deepcopy(nodeDict)  
    capacityDictCopy = copy.deepcopy(capacityDict)
    paths = []
    pathFlows = []
    overallCost = 0
     # 需要运送的流量
    givenFlow = 70
    eps = 1e-5
    # 开始循环
    while nodeDictCopy['1'] and (abs(givenFlow) > eps):
        # 生成路径
        path = genPath(ind, nodeDictCopy)  
        # 当路径无法抵达终点，说明经过这个节点已经无法往下走，从所有其他节点的指向中删除该节点
        if path[-1] != geneLength:
             for node, nextNode in nodeDictCopy.items():
                 if path[-1] in nextNode:
                     nodeDictCopy[node].remove(path[-1])
             continue
             # 保存路径
        paths.append(path)
        # 计算路径最大流量
        pathFlow = traceCapacity(path, capacityDictCopy) 
        # 当剩余流量不能填满该路径的最大流量时，将所有剩余流量分配给该路径
        if givenFlow < pathFlow: 
            pathFlow = givenFlow     
            # 保存路径的流量       
        pathFlows.append(pathFlow) 
        # 更新需要运送的剩余流量
        givenFlow -= pathFlow 
        # 更新节点链表
        updateNodeDict(capacityDictCopy, nodeDictCopy) 
        # 计算路径上的cost
        pathCost = calCost(path, pathFlow, costDict)
        overallCost += pathCost        
    if outputPaths:
        return overallCost, paths, pathFlows
    return overallCost,
toolbox.register('evaluate', evaluate)

# 迭代数据
stats = tools.Statistics(key=lambda ind:ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)

# 生成初始族群
toolbox.popSize = 100
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(toolbox.popSize)

# 注册工具
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.5)

# --------------------
# 遗传算法参数
toolbox.ngen = 200
toolbox.cxpb = 0.8
toolbox.mutpb = 0.05

# 遗传算法主程序部分
hallOfFame = tools.HallOfFame(maxsize=1)
pop,logbook = algorithms.eaSimple(pop, toolbox, cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
                   ngen = toolbox.ngen, stats=stats, halloffame=hallOfFame, verbose=True)

# 输出结果
from pprint import pprint
bestInd = hallOfFame.items[0]
overallCost, paths, pathFlows = eval(bestInd, outputPaths=True)
print('最优解路径为: ')
pprint(paths)
print('各路径上的流量为：'+str(pathFlows))
print('最小费用为: '+str(overallCost))

# 可视化迭代过程
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()

## 结果：
#最优解路径为: 
#[[1, 2, 7, 13, 17, 22, 25],
# [1, 2, 8, 14, 19, 20, 23, 25],
# [1, 3, 7, 13, 17, 22, 25],
# [1, 3, 9, 13, 17, 22, 25],
# [1, 3, 8, 14, 19, 23, 25],
# [1, 4, 9, 13, 17, 22, 25],
# [1, 4, 9, 13, 18, 22, 25],
# [1, 4, 8, 14, 19, 23, 25],
# [1, 4, 8, 12, 17, 22, 25],
# [1, 4, 10, 16, 20, 23, 25],
# [1, 4, 10, 16, 20, 24, 25],
# [1, 5, 9, 13, 18, 19, 23, 25],
# [1, 5, 9, 15, 20, 24, 25],
# [1, 5, 11, 15, 20, 24, 25]]
#各路径上的流量为：[10, 8, 5, 4, 5, 1, 7, 2, 3, 2, 5, 3, 7, 8]
#最小费用为: 6971

# https://www.jianshu.com/p/1d09ef7a1f27
