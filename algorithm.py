import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import WCCI2020MTSO as WCCI
import CEC2017MTSO as CEC
import population_operation_tmp as po
import operator
import gauss

MaxFEs = 100000
G = 1000  # 迭代次数
k = 6  # 交叉方式选择SBX交叉
NP = 50  # 个体数目


# 算法本身
def algorithm(task1, task2, question, alpha_p=0.1, beta_p=0.9, reward_p=10, g_p=5, ilp=0.5):
    aaa = alpha_p
    bbb = beta_p
    reward = reward_p
    gg = g_p
    rmp = ilp
    if question == 'PILS':
        maxD1 = 50
        maxD2 = 25
    else:
        maxD1 = 50
        maxD2 = 50
    # 初始化种群
    population1, population2 = po.t_init(maxD1, maxD2)
    # 对种群个体进行初始评价
    population1, t1_gbest_F, t1_gbest_X = po.t_evaluate(population1, task1)
    population2, t2_gbest_F, t2_gbest_X = po.t_evaluate(population2, task2)
    # 对于每一个种群，记录它当前的任务的适应值
    for i in range(NP):
        population1[i]['cost1'] = population1[i]['cost']
        population1[i]['cost2'] = np.inf
        population2[i]['cost2'] = population2[i]['cost']
        population2[i]['cost1'] = np.inf
    t1 = []  # 记录任务一每一代最优
    t2 = []  # 记录任务二每一代最优
    Qtable1 = 50 * np.ones((maxD1, maxD1 + maxD2))  # 任务1Q表
    Qtable2 = 50 * np.ones((maxD2, maxD2 + maxD1))  # 任务2Q表
    fes = 0
    g = 0
    # 进行迭代
    while fes < MaxFEs:
        population1 = gauss.get_P(population1, 1)
        population2 = gauss.get_P(population2, 2)
        # 遍历种群1
        for i in range(NP):
            # 信息交流
            if g % gg == 0 and g != 0 and random.uniform(0, 1) < rmp:
                # 选另一个任务的一个个体作为不平衡学习对象
                index = random.randint(0, NP - 1)
                # 形成方案
                d = {}
                d['plan'] = []  # 方案
                d['information'] = np.zeros(maxD1)  # 策略
                d['child'] = True  # 标记为迁移后代
                for n in range(maxD1):
                    P = Qtable1[n] / np.sum(Qtable1[n] + 1e-10)  # 求第n维，像所有维度学习的概率
                    indexD = np.random.choice(range(maxD1 + maxD2), p=P)  # 根据概率选择维度
                    if indexD < maxD1:
                        d['plan'].append(indexD)
                        d['information'][n] = population1[i]['information'][indexD]
                    else:
                        d['plan'].append(indexD)
                        d['information'][n] = population2[index]['information'][indexD - maxD1]
                indexs = [index for index in range(NP) if index != i]
                d = po.t_DE_current(population1[i], d, 1, maxD1)
            else:
                if random.uniform(0, 1) < 1 - population1[i]['P']:
                    indexs = [index for index in range(NP) if index != i]
                    r1, r2, r3 = np.random.choice(indexs, 3, replace=False)  # 选3个不相同的下标
                    d = po.t_DE(population1[r1], population1[r2], population1[r3], population1[i], maxD1)
                else:
                    d = {}
                    d['child'] = False
                    d['information'] = gauss.local_search(population1[i]['information'], fes, maxD1)
            d['cost1'] = task1.function(d["information"])
            d['cost2'] = np.inf
            population1.append(d)
            fes += 1
        # 遍历种群2
        for i in range(NP):
            if g % gg == 0 and g != 0 and random.uniform(0, 1) < rmp:
                # 选另一个任务的一个个体作为不平衡学习对象
                index = random.randint(0, NP - 1)
                # 形成方案
                d = {}
                d['plan'] = []  # 方案
                d['information'] = np.zeros(maxD2)  # 策略
                d['child'] = True  # 标记为迁移后代
                for n in range(maxD2):
                    P = Qtable2[n] / np.sum(Qtable2[n] + 1e-10)  # 求第n维，像所有维度学习的概率
                    indexD = np.random.choice(range(maxD2 + maxD1), p=P)  # 根据概率选择维度
                    if indexD < maxD2:
                        d['plan'].append(indexD)
                        d['information'][n] = population2[i]['information'][indexD]
                    else:
                        d['plan'].append(indexD)
                        d['information'][n] = population1[index]['information'][indexD - maxD2]
                indexs = [index for index in range(NP) if index != i]
                d = po.t_DE_current(population2[i], d, 2, maxD2)
            else:
                if random.uniform(0, 1) < 1 - population2[i]['P']:
                    indexs = [index for index in range(NP) if index != i]
                    r1, r2, r3 = np.random.choice(indexs, 3, replace=False)  # 选3个不相同的下标
                    d = po.t_DE(population2[r1], population2[r2], population2[r3], population2[i], maxD2)
                else:
                    d = {}
                    d['child'] = False
                    d['information'] = gauss.local_search(population2[i]['information'], fes, maxD2)
            d['cost2'] = task2.function(d["information"])
            d['cost1'] = np.inf
            population2.append(d)
            fes += 1
        # 当代个体生成完毕，且适应值求解完成，开始评价
        # 根据因子代价1排序
        population1 = sorted(population1, key=operator.itemgetter('cost1'))
        population2 = sorted(population2, key=operator.itemgetter('cost2'))
        # 更新Q表
        for i in range(NP):
            if population1[i]['child']:
                if population1[i]['cost1'] < population1[i]['parent_cost']:
                    for n in range(maxD1):
                        maxQ = max(Qtable1[n])
                        Qtable1[n][population1[i]['plan'][n]] += aaa * (
                                reward * 2 + bbb * maxQ - Qtable1[n][population1[i]['plan'][n]])
                else:
                    for n in range(maxD1):
                        maxQ = max(Qtable1[n])
                        Qtable1[n][population1[i]['plan'][n]] += aaa * (
                                reward + bbb * maxQ - Qtable1[n][population1[i]['plan'][n]])
            if population2[i]['child']:
                if population2[i]['cost2'] < population2[i]['parent_cost']:
                    for n in range(maxD2):
                        maxQ = max(Qtable2[n])
                        Qtable2[n][population2[i]['plan'][n]] += aaa * (
                                reward * 2 + bbb * maxQ - Qtable2[n][population2[i]['plan'][n]])
                else:
                    for n in range(maxD2):
                        maxQ = max(Qtable2[n])
                        Qtable2[n][population2[i]['plan'][n]] += aaa * (
                                reward + bbb * maxQ - Qtable2[n][population2[i]['plan'][n]])

        # 删除后面部分,2NP个个体只保留前NP个个体
        for i in range(NP):
            population1[i]['child'] = False
            population2[i]['child'] = False
            population1.pop()
            population2.pop()

        # 更新最优
        for i in range(NP):
            if t1_gbest_F > population1[i]["cost1"]:  # 更新任务1最优
                t1_gbest_F = population1[i]["cost1"]
                t1_gbest_X = population1[i]["information"].copy()
            if t2_gbest_F > population2[i]["cost2"]:  # 更新任务2最优
                t2_gbest_F = population2[i]["cost2"]
                t2_gbest_X = population2[i]["information"].copy()
        # 记录当前代最优
        t1.append(t1_gbest_F)
        t2.append(t2_gbest_F)
        g += 1
        if g % 100 == 0:
            print(g, t1_gbest_F, t2_gbest_F)  # 输出当前代两个任务的最优值
    print(question, t1_gbest_F, t2_gbest_F)  # 输出question问题下两个任务的最优值

    return t1, t2, t1_gbest_F, t2_gbest_F
