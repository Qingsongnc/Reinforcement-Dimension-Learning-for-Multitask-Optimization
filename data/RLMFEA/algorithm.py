import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import WCCI2020MTSO as WCCI
import CEC2017MTSO as CEC
import population_operation as po
import operator

G = 1000  # 迭代次数
k = 6  # 交叉方式选择SBX交叉
NP = 100  # 个体数目
maxD = 50  # 所有问题的最大维度

Q1 = [[0 for i in range(7)] for j in range(3)]  # 强化学习Q表
Q2 = [[0 for i in range(7)] for j in range(3)]  # 强化学习Q表
aaa = 0.1  # 学习率
bbb = 0.9  # 折扣率


# 算法本身
def algorithm(task1, task2, question):
    rmp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    rmp1_index = 2  # 初始设置rmp为0.3
    rmp2_index = 2  # 初始设置rmp为0.3
    rmp1 = rmp[rmp1_index]  # 交叉概率
    rmp2 = rmp[rmp2_index]  # 交叉概率
    state_now1 = 0  # 当前状态对应Q1表格的行
    state_next1 = -1  # 下一状态对应Q1表格的行
    state_now2 = 0  # 当前状态对应Q2表格的行
    state_next2 = -1  # 下一状态对应Q1表格的行
    # 初始化种群
    population = po.f_init()
    # 对种群个体进行初始评价
    population, t1_gbest_F, t1_gbest_X, t2_gbest_F, t2_gbest_X = po.f_evaluate(population, task1, task2)
    t1 = []  # 记录任务一每一代最优
    t2 = []  # 记录任务二每一代最优
    # 进行迭代
    for g in range(G):
        # 分配算子，0为DE，1为GA
        rand = random.uniform(0, 1)
        if rand <= 0.5:
            # 任务1-技能因子1-DE 任务2-技能因子2-GA
            S1, S2 = 'DE', 'GA'
        else:
            # 任务1-技能因子1-GA 任务2-技能因子2-DE
            S1, S2 = 'GA', 'DE'
        # 遍历每一个任务
        for s in range(1, 3):
            # 遍历每一个个体
            for i in range(NP):
                # 技能因子跟当前任务不一致，跳过
                if population[i]['skill'] != s:
                    continue
                # 随机选择一个技能因子为当前技能因子的个体产生后代
                r1 = random.randint(0, NP - 1)
                while population[r1]['skill'] != s or i == r1:
                    r1 = random.randint(0, NP - 1)
                # 确定是否知识迁移
                rand = random.uniform(0, 1)
                if rand <= rmp1:
                    # 选技能因子不同的
                    indexs = [index for index in range(NP) if
                              index != i and population[i]['skill'] != population[index]['skill']]
                    r2, r3 = np.random.choice(indexs, 2, replace=False)  # 选2个不相同的下标
                else:
                    # 选技能因子相同的
                    indexs = [index for index in range(NP) if
                              index != i and population[i]['skill'] == population[index]['skill']]
                    r2, r3 = np.random.choice(indexs, 2, replace=False)  # 选2个不相同的下标
                if s == 1:
                    if S1 == 'DE':
                        d = po.t_DE(population[r1], population[r2], population[r3], population[i])
                        if rand <= rmp1:
                            rand = random.uniform(0, 1)
                            if rand <= 0.5:
                                d['skill'] = 1
                            else:
                                d['skill'] = 2
                        else:
                            d['skill'] = 1
                    elif S1 == 'GA':
                        d1, d2 = po.crossover(population, r1, i, 6)
                        rand = random.uniform(0, 1)
                        if rand < 0.5:
                            d = d1
                        else:
                            d = d2
                        d = po.mutated(d)
                if s == 2:
                    if S2 == 'DE':
                        d = po.t_DE(population[r1], population[r2], population[r3], population[i])
                        if rand <= rmp2:
                            rand = random.uniform(0, 1)
                            if rand <= 0.5:
                                d['skill'] = 2
                            else:
                                d['skill'] = 1
                        else:
                            d['skill'] = 2
                    elif S2 == 'GA':
                        d1, d2 = po.crossover(population, r1, i, 6)
                        rand = random.uniform(0, 1)
                        if rand < 0.5:
                            d = d1
                        else:
                            d = d2
                        d = po.mutated(d)
                # 求适应值
                # 计算这两个的适应值
                if d['skill'] == 1:
                    d['cost1'] = task1.function(d['information'])
                    d['cost2'] = np.inf
                else:
                    d['cost1'] = np.inf
                    d['cost2'] = task2.function(d['information'])
                # 加入种群
                population.append(d)
        # 确认状态
        pbest1 = np.inf
        pbest2 = np.inf
        cbest1 = np.inf
        cbest2 = np.inf
        for i in range(NP):
            if population[i]['skill'] == 1 and population[i]['cost1'] < pbest1:
                pbest1 = population[i]['cost1']
            if population[i]['skill'] == 2 and population[i]['cost2'] < pbest2:
                pbest2 = population[i]['cost2']
        for i in range(NP, 2 * NP):
            if population[i]['skill'] == 1 and population[i]['cost1'] < cbest1:
                cbest1 = population[i]['cost1']
            if population[i]['skill'] == 2 and population[i]['cost2'] < cbest2:
                cbest2 = population[i]['cost2']
        # 确认回报和下一状态
        if cbest1 < pbest1:
            Reward1, state_next1 = 10, 0
        elif abs(cbest1 - pbest1) < 0.0000001:
            Reward1, state_next1 = 5, 1
        elif cbest1 > pbest1:
            Reward1, state_next1 = 0, 2
        if cbest2 < pbest2:
            Reward2, state_next2 = 10, 0
        elif abs(cbest2 - pbest2) < 0.0000001:
            Reward2, state_next2 = 5, 1
        elif cbest2 > pbest2:
            Reward2, state_next2 = 0, 2
        # 更新Q1表
        i = state_now1
        j = rmp1_index
        maxQ = max(Q1[state_next1])
        Q1[i][j] = Q1[i][j] + aaa * (Reward1 + bbb * maxQ - Q1[i][j])
        # 更新Q2表
        i = state_now2
        j = rmp2_index
        maxQ = max(Q2[state_next2])
        Q2[i][j] = Q2[i][j] + aaa * (Reward2 + bbb * maxQ - Q2[i][j])
        # 更新状态
        state_now1 = state_next1
        state_now2 = state_next2
        # 确认rmp1和rmp2
        w = 1
        eee = [math.e ** Q1[state_now1][i] / w for i in range(7)]
        P = eee / np.sum(eee)
        rmp1_index = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1, p=P)
        rmp1_index = int(rmp1_index)
        rmp1 = rmp[rmp1_index]

        w = 1
        eee = [math.e ** Q2[state_now2][i] / w for i in range(7)]
        P = eee / np.sum(eee)
        rmp2_index = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1, p=P)
        rmp2_index = int(rmp2_index)
        rmp2 = rmp[rmp2_index]

        # 当代个体生成完毕，且适应值求解完成，开始评价
        # 根据因子代价1排序
        population = sorted(population, key=operator.itemgetter('cost1'))
        # 评价因子排名1
        for i in range(2 * NP):
            population[i]["rank1"] = i + 1
        # 根据因子代价2排序
        population = sorted(population, key=operator.itemgetter('cost2'))
        # 评价因子排名2
        for i in range(2 * NP):
            population[i]["rank2"] = i + 1
        # 评价适应值标量
        for i in range(2 * NP):
            if population[i]["rank1"] < population[i]["rank2"]:
                population[i]["fitness"] = 1 / population[i]["rank1"]
            else:
                population[i]["fitness"] = 1 / population[i]["rank2"]
        # 根据适应值标量排序
        population = sorted(population, key=operator.itemgetter('fitness'), reverse=True)
        # 删除后面部分,2NP个个体只保留前NP个个体
        for i in range(NP):
            population.pop()
        # 更新最优
        for i in range(NP):
            if t1_gbest_F > population[i]["cost1"]:  # 更新任务1最优
                t1_gbest_F = population[i]["cost1"]
                t1_gbest_X = population[i]["information"].copy()
            if t2_gbest_F > population[i]["cost2"]:  # 更新任务2最优
                t2_gbest_F = population[i]["cost2"]
                t2_gbest_X = population[i]["information"].copy()

        # 记录当前代最优
        t1.append(t1_gbest_F)
        t2.append(t2_gbest_F)
        if g % 10 == 0:
            print(g, t1_gbest_F, t2_gbest_F)  # 输出当前代两个任务的最优值
    print(question, t1_gbest_F, t2_gbest_F)  # 输出question问题下两个任务的最优值
    return t1, t2, t1_gbest_F, t2_gbest_F
