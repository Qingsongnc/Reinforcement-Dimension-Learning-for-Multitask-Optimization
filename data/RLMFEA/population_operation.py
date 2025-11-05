import random
import numpy as np
import WCCI2020MTSO as WCCI
import CEC2017MTSO as CEC
import operator

# 此为一般种群操作
NP = 100  # 个体数目
maxD = 50  # 所有问题的最大维度
# GA
lameda = 0.25  # 算术交叉的参数
oumeiga = 0.25  # 几何交叉的参数
aerfa = 0.3  # BLX-alpha的参数
mu = 10  # SBX交叉的参数
mum = 5  # 多项式变异的参数
# DE
F = 0.5  # 缩放因子
Cr = 0.6  # 交叉概率


# 单种群初始化，随机生成NP个个体，每个个体的维数为所有任务中最大的那个
def f_init():
    population = []
    for i in range(NP):
        tmp = {}
        tmp["information"] = np.random.uniform(low=0, high=1, size=[maxD])
        population.append(tmp)
    return population


# 对于初始单种群population，就task1和task2问题，进行初步评价,计算技能因子，返回最新种群信息和最优信息
def f_evaluate(population, task1, task2):
    t1_gbest_F = np.inf  # 任务1的最优值
    t1_gbest_X = []  # 任务1的最优个体
    t2_gbest_F = np.inf  # 任务2的最优值
    t2_gbest_X = []  # 任务2的最优个体
    # 评价因子代价cost1和cost2
    for i in range(NP):
        for task in range(1, 3):
            if task == 1:
                population[i]["cost1"] = task1.function(population[i]["information"])
            elif task == 2:
                population[i]["cost2"] = task2.function(population[i]["information"])
    # 根据因子代价1排序
    population = sorted(population, key=operator.itemgetter('cost1'))
    # 评价因子排名1
    for i in range(NP):
        population[i]["rank1"] = i + 1
    # 根据因子代价2排序
    population = sorted(population, key=operator.itemgetter('cost2'))
    # 评价因子排名2
    for i in range(NP):
        population[i]["rank2"] = i + 1
    # 评价适应值标量和技能因子
    for i in range(NP):
        # 在任务一排名靠前，技能因子为1
        if population[i]["rank1"] < population[i]["rank2"]:
            population[i]["fitness"] = 1 / population[i]["rank1"]
            population[i]["skill"] = 1
            if t1_gbest_F > population[i]["cost1"]:  # 更新任务1最优
                t1_gbest_F = population[i]["cost1"]
                t1_gbest_X = population[i]["information"].copy()
        # 在任务二排名靠前，技能因子为2
        elif population[i]["rank1"] > population[i]["rank2"]:
            population[i]["fitness"] = 1 / population[i]["rank2"]
            population[i]["skill"] = 2
            if t2_gbest_F > population[i]["cost2"]:  # 更新任务2最优
                t2_gbest_F = population[i]["cost2"]
                t2_gbest_X = population[i]["information"].copy()
        # 如果排名一样，随机选择技能因子
        elif population[i]["rank1"] == population[i]["rank2"]:
            population[i]["fitness"] = 1 / population[i]["rank2"]
            population[i]["skill"] = random.randint(1, 2)
            if population[i]["skill"] == 1:
                if t1_gbest_F > population[i]["cost1"]:  # 更新任务1最优
                    t1_gbest_F = population[i]["cost1"]
                    t1_gbest_X = population[i]["information"].copy()
            elif population[i]["skill"] == 2:
                if t2_gbest_F > population[i]["cost2"]:  # 更新任务2最优
                    t2_gbest_F = population[i]["cost2"]
                    t2_gbest_X = population[i]["information"].copy()
    return population, t1_gbest_F, t1_gbest_X, t2_gbest_F, t2_gbest_X


# 多种群初始化，随机生成NP个个体，每个个体的维数为所有任务中最大的那个
def t_init():
    pop1 = []
    for i in range(NP):
        tmp = {}
        tmp["information"] = np.random.uniform(low=0, high=1, size=[maxD])
        pop1.append(tmp)
    pop2 = []
    for i in range(NP):
        tmp = {}
        tmp["information"] = np.random.uniform(low=0, high=1, size=[maxD])
        pop2.append(tmp)
    return pop1, pop2


# 对于初始多种群population，就task问题，进行初步评价,计算适应值，返回最新种群信息和最优信息
def t_evaluate(population, task):
    t1_gbest_F = np.inf  # 任务的最优值
    t1_gbest_X = []  # 任务的最优个体
    # 评价因子代价cost
    for i in range(NP):
        population[i]["cost"] = task.function(population[i]["information"])
        if t1_gbest_F > population[i]["cost"]:  # 更新任务1最优
            t1_gbest_F = population[i]["cost"]
            t1_gbest_X = population[i]["information"].copy()
    return population, t1_gbest_F, t1_gbest_X


# population种群中的i,j个体以k方式交叉,其中6为模拟二进制交叉SBX
def crossover(population, i, j, k):
    c1 = []  # 新个体1的information
    c2 = []  # 新个体2的information
    if k == 1:  # 两点交叉
        indexs = [index for index in range(maxD)]
        r1, r2 = np.random.choice(indexs, 2, replace=False)  # 选2个不相同的下标
        if r1 > r2:
            r1, r2 = r2, r1
        for x in range(maxD):
            if x < r1 or x > r2:
                c1.append(population[i]["information"][x])
            else:
                c1.append(population[j]["information"][x])
        r1, r2 = np.random.choice(indexs, 2, replace=False)  # 选2个不相同的下标
        if r1 > r2:
            r1, r2 = r2, r1
        for x in range(maxD):
            if x < r1 or x > r2:
                c2.append(population[j]["information"][x])
            else:
                c2.append(population[i]["information"][x])
    elif k == 2:  # 均匀交叉
        indexs = [0, 1]
        for x in range(maxD):
            u = np.random.choice(indexs, 1, replace=False)  # 随机选择0或者1
            if u == 0:
                c1.append(population[i]["information"][x])
            elif u == 1:
                c1.append(population[j]["information"][x])
            u = np.random.choice(indexs, 1, replace=False)  # 随机选择0或者1
            if u == 0:
                c2.append(population[i]["information"][x])
            elif u == 1:
                c2.append(population[j]["information"][x])
    elif k == 3:  # 算数交叉
        for x in range(maxD):
            c1.append(lameda * population[i]["information"][x] + (1 - lameda) * population[j]["information"][x])
            c2.append(lameda * population[j]["information"][x] + (1 - lameda) * population[i]["information"][x])
    elif k == 4:  # 几何交叉
        for x in range(maxD):
            c1.append((population[i]["information"][x] ** oumeiga) * (population[j]["information"][x] ** (1 - oumeiga)))
            c2.append((population[j]["information"][x] ** oumeiga) * (population[i]["information"][x] ** (1 - oumeiga)))
    elif k == 5:  # BLX-alpha交叉
        for x in range(maxD):
            Pmax = max(population[i]["information"][x], population[j]["information"][x])
            Pmin = min(population[i]["information"][x], population[j]["information"][x])
            I = Pmax - Pmin
            c1.append(random.uniform(Pmin - I * aerfa, Pmax + I * aerfa))
            c2.append(random.uniform(Pmin - I * aerfa, Pmax + I * aerfa))
    elif k == 6:  # SBX交叉
        u = np.random.uniform(low=0, high=1, size=[maxD])
        cf = np.zeros(maxD)
        for x in range(maxD):
            if u[x] <= 0.5:
                cf[x] = (2 * u[x]) ** (1 / (mu + 1))
            else:
                cf[x] = (2 * (1 - u[x])) ** (-1 / (mu + 1))
            tmp = 0.5 * ((1 - cf[x]) * population[i]["information"][x] + (1 + cf[x]) * population[j]["information"][x])
            c1.append(tmp)
            tmp = 0.5 * ((1 + cf[x]) * population[i]["information"][x] + (1 - cf[x]) * population[j]["information"][x])
            c2.append(tmp)
    c1 = np.array(c1)  # 列表转nparray
    c2 = np.array(c2)  # 列表转nparray
    # 控边界
    # 将大于1的元素设为1，小于0的元素设为0
    c1[c1 > 1] = 1
    c1[c1 < 0] = 0
    c2[c2 > 1] = 1
    c2[c2 < 0] = 0
    d1 = {}  # 新个体1的整体信息
    d2 = {}  # 新个体2的整体信息
    d1["information"] = c1
    d2["information"] = c2
    # 垂直文化传播
    if random.uniform(0, 1) < 0.5:
        d1["skill"] = population[i]["skill"]
    else:
        d1["skill"] = population[j]["skill"]
    if random.uniform(0, 1) < 0.5:
        d2["skill"] = population[i]["skill"]
    else:
        d2["skill"] = population[j]["skill"]
    return d1, d2

# population种群中的i,j个体以k方式交叉,其中6为模拟二进制交叉SBX（多种群）
def t_crossover(population, i, j, k):
    c1 = []  # 新个体1的information
    c2 = []  # 新个体2的information
    if k == 1:  # 两点交叉
        indexs = [index for index in range(maxD)]
        r1, r2 = np.random.choice(indexs, 2, replace=False)  # 选2个不相同的下标
        if r1 > r2:
            r1, r2 = r2, r1
        for x in range(maxD):
            if x < r1 or x > r2:
                c1.append(population[i]["information"][x])
            else:
                c1.append(population[j]["information"][x])
        r1, r2 = np.random.choice(indexs, 2, replace=False)  # 选2个不相同的下标
        if r1 > r2:
            r1, r2 = r2, r1
        for x in range(maxD):
            if x < r1 or x > r2:
                c2.append(population[j]["information"][x])
            else:
                c2.append(population[i]["information"][x])
    elif k == 2:  # 均匀交叉
        indexs = [0, 1]
        for x in range(maxD):
            u = np.random.choice(indexs, 1, replace=False)  # 随机选择0或者1
            if u == 0:
                c1.append(population[i]["information"][x])
            elif u == 1:
                c1.append(population[j]["information"][x])
            u = np.random.choice(indexs, 1, replace=False)  # 随机选择0或者1
            if u == 0:
                c2.append(population[i]["information"][x])
            elif u == 1:
                c2.append(population[j]["information"][x])
    elif k == 3:  # 算数交叉
        for x in range(maxD):
            c1.append(lameda * population[i]["information"][x] + (1 - lameda) * population[j]["information"][x])
            c2.append(lameda * population[j]["information"][x] + (1 - lameda) * population[i]["information"][x])
    elif k == 4:  # 几何交叉
        for x in range(maxD):
            c1.append((population[i]["information"][x] ** oumeiga) * (population[j]["information"][x] ** (1 - oumeiga)))
            c2.append((population[j]["information"][x] ** oumeiga) * (population[i]["information"][x] ** (1 - oumeiga)))
    elif k == 5:  # BLX-alpha交叉
        for x in range(maxD):
            Pmax = max(population[i]["information"][x], population[j]["information"][x])
            Pmin = min(population[i]["information"][x], population[j]["information"][x])
            I = Pmax - Pmin
            c1.append(random.uniform(Pmin - I * aerfa, Pmax + I * aerfa))
            c2.append(random.uniform(Pmin - I * aerfa, Pmax + I * aerfa))
    elif k == 6:  # SBX交叉
        u = np.random.uniform(low=0, high=1, size=[maxD])
        cf = np.zeros(maxD)
        for x in range(maxD):
            if u[x] <= 0.5:
                cf[x] = (2 * u[x]) ** (1 / (mu + 1))
            else:
                cf[x] = (2 * (1 - u[x])) ** (-1 / (mu + 1))
            tmp = 0.5 * ((1 - cf[x]) * population[i]["information"][x] + (1 + cf[x]) * population[j]["information"][x])
            c1.append(tmp)
            tmp = 0.5 * ((1 + cf[x]) * population[i]["information"][x] + (1 - cf[x]) * population[j]["information"][x])
            c2.append(tmp)
    c1 = np.array(c1)  # 列表转nparray
    c2 = np.array(c2)  # 列表转nparray
    # 控边界
    # 将大于1的元素设为1，小于0的元素设为0
    c1[c1 > 1] = 1
    c1[c1 < 0] = 0
    c2[c2 > 1] = 1
    c2[c2 < 0] = 0
    d1 = {}  # 新个体1的整体信息
    d2 = {}  # 新个体2的整体信息
    d1["information"] = c1
    d2["information"] = c2
    return d1, d2
# array1个体多项式变异
def mutated(array1):
    c = []
    d = {}
    for x in range(maxD):
        rand = random.uniform(0, 1)
        if rand < 1 / maxD:  # 以1/maxD的概率变异
            u = random.uniform(0, 1)
            if u <= 0.5:  # 以第一个公式变异
                tmp1 = (2 * u) ** (1 / (1 + mum)) - 1
                tmp2 = array1["information"][x] + tmp1 * array1["information"][x]
                c.append(tmp2)
            else:  # 以第二个公式变异
                tmp1 = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
                tmp2 = array1["information"][x] + tmp1 * (1 - array1["information"][x])
                c.append(tmp2)
        else:  # 不变异
            c.append(array1["information"][x])
    c = np.array(c)
    # 控边界
    c[c > 1] = 1
    c[c < 0] = 0
    d["information"] = c
    d["skill"] = array1["skill"]
    return d
# array1个体多项式变异(多种群)
def t_mutated(array1):
    c = []
    d = {}
    for x in range(maxD):
        rand = random.uniform(0, 1)
        if rand < 1 / maxD:  # 以1/maxD的概率变异
            u = random.uniform(0, 1)
            if u <= 0.5:  # 以第一个公式变异
                tmp1 = (2 * u) ** (1 / (1 + mum)) - 1
                tmp2 = array1["information"][x] + tmp1 * array1["information"][x]
                c.append(tmp2)
            else:  # 以第二个公式变异
                tmp1 = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
                tmp2 = array1["information"][x] + tmp1 * (1 - array1["information"][x])
                c.append(tmp2)
        else:  # 不变异
            c.append(array1["information"][x])
    c = np.array(c)
    # 控边界
    c[c > 1] = 1
    c[c < 0] = 0
    d["information"] = c
    return d

# 单种群DE/rand/1
def f_DE(population, i, r1, r2, r3):
    v = population[r1]['information'] + F * (population[r2]['information'] - population[r3]['information'])
    # 保证解的有效性
    v[v < 0] = 0
    v[v > 1] = 1
    # 交叉操作
    krand = random.randint(0, maxD - 1)  # jrand
    rand = np.random.uniform(low=0, high=1, size=[maxD])  # rand
    u = population[i]['information'].copy()
    for k in range(maxD):
        if rand[k] <= Cr or k == krand:
            u[k] = v[k]
    d = {}
    d['information'] = u.copy()
    if population[r1]['skill'] == population[r2]['skill']:
        d['skill'] = population[r1]['skill']
    else:
        rand = random.uniform(0, 1)
        if rand < 0.5:
            d['skill'] = population[r1]['skill']
        else:
            d['skill'] = population[r2]['skill']
    return d


# 多种群DE/rand/1
def t_DE(array1, array2, array3, array4):
    # 执行变异操作
    v = array1['information'] + F * (array2['information'] - array3['information'])
    # 保证解的有效性
    v[v < 0] = 0
    v[v > 1] = 1
    # 交叉操作
    krand = random.randint(0, maxD - 1)  # jrand
    rand = np.random.uniform(low=0, high=1, size=[maxD])  # rand
    u = array4['information'].copy()
    for k in range(maxD):
        if rand[k] <= Cr or k == krand:
            u[k] = v[k]
    d = {}
    d['information'] = u.copy()
    return d
