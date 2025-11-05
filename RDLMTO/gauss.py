import random
import numpy as np
NP=50
MaxFEs = 100000
seita=0.0001
#高斯分布
# def Gauss(array,sigma,maxD):
#     a=[]
#     for i in range(maxD):
#         a.append(random.gauss(array[i],sigma))
#     a=np.array(a)
#     return a.copy()

# 基于高斯的局部搜索策略
def local_search(array, fes,maxD):

    gamma = 1 + fes  / MaxFEs * 4
    sigma = 10 ** (-gamma)
    d = {}
    d['information'] = np.random.uniform(low=0, high=1, size=[maxD])
    for j in range(maxD):
        d['information'][j] = random.gauss(mu=array[j], sigma=sigma)
        if d['information'][j] > 1:
            d['information'][j] = 1
        if d['information'][j] < 0:
            d['information'][j] = 0
    return d['information'].copy()


def get_P(pop,index):
    max=-np.inf
    min=np.inf
    for i in range(NP):
        if index==1:
            if pop[i]['cost1']>max:
                max=pop[i]['cost1']
            if pop[i]['cost1']<min:
                min=pop[i]['cost1']
        else:
            if pop[i]['cost2']>max:
                max=pop[i]['cost2']
            if pop[i]['cost2']<min:
                min=pop[i]['cost2']
    for i in range(NP):
        if index==1:
            pop[i]['P']=1-(pop[i]['cost1']-min+seita)/(max-min+seita)
        else:
            pop[i]['P']=1-(pop[i]['cost2']-min+seita)/(max-min+seita)
    return pop