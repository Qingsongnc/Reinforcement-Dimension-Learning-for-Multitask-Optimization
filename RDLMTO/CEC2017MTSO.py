import scipy.io as sio
import numpy as np
import math


# 此为CEC2017MTSO测试集，使用方法为：
# task1=Tasks('CIHS',1)
# task2=Tasks('CIHS',2)
# fitness=task1.function(array)

# 基本函数
def Sphere(array, opt):
    array = array - opt
    return np.sum(array ** 2)


def Rosenbrock(array):
    return np.sum(100 * (array[1:] - array[:-1] ** 2) ** 2 + (1 - array[:-1]) ** 2)


def Ackley(array, M, opt):
    d = len(array)
    array = array - opt
    array = np.dot(M, array.T)
    array = array.T
    res = - 20 * np.exp(-0.2 * np.sqrt(np.mean(array ** 2)))
    res = res - np.exp(np.mean(np.cos(2 * np.pi * array))) + 20 + np.exp(1)
    return res


def Rastrgin(array, M, opt):
    d = len(array)
    array = array - opt
    array = np.dot(M, array.T)
    array = array.T
    return 10 * d + np.sum(array ** 2 - 10 * np.cos(2 * np.pi * array))


def Griewank(array, M, opt):
    d = len(array)
    array = array - opt
    array = np.dot(M, array.T)
    array = array.T
    i = np.arange(1, d + 1)
    return 1 + np.sum(array ** 2) / 4000 - np.prod(np.cos(array / np.sqrt(i)))


def Weierstrass(array, M, opt):
    d = len(array)
    array = array - opt
    array = np.dot(M, array.T)
    array = array.T
    array = array[0]
    a = 0.5
    b = 3
    kmax = 20
    result = 0
    for i in range(d):
        tmp1 = 0
        for k in range(kmax + 1):
            tmp1 += (a ** k) * math.cos(2 * math.pi * (b ** k) * (array[i] + 0.5))
        result += tmp1
    for k in range(kmax + 1):
        result -= d * (a ** k) * math.cos(0.5 * 2 * math.pi * (b ** k))
    return result


def Schwefel(array):
    d = len(array)
    return 418.9829 * d - np.sum(array * np.sin(np.sqrt(np.abs(array))))


# 九个基本问题
def CIHS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -100, 100, 50
        x = l + array * (u - l)
        x = x[:d]
        return Griewank(x, M, opt)
    elif task == 2:
        M = data["Rotation_Task2"]
        opt = data["GO_Task2"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Rastrgin(x, M, opt)


def CIMS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        M = data["Rotation_Task2"]
        opt = data["GO_Task2"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Rastrgin(x, M, opt)


def CILS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        l, u, d = -500, 500, 50
        x = l + array * (u - l)
        x = x[:d]
        return Schwefel(x)


def PIHS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Rastrgin(x, M, opt)
    elif task == 2:
        opt = data["GO_Task2"]
        l, u, d = -100, 100, 50
        x = l + array * (u - l)
        x = x[:d]
        return Sphere(x, opt)


def PIMS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Rosenbrock(x)


def PILS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Ackley(x, M, opt)
    elif task == 2:
        M = data["Rotation_Task2"]
        opt = data["GO_Task2"]
        l, u, d = -0.5, 0.5, 25
        x = l + array * (u - l)
        x = x[:d]
        return Weierstrass(x, M, opt)


def NIHS(array, task, data):
    if task == 1:
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Rosenbrock(x)
    elif task == 2:
        M = data["Rotation_Task2"]
        opt = data["GO_Task2"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Rastrgin(x, M, opt)


def NIMS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -100, 100, 50
        x = l + array * (u - l)
        x = x[:d]
        return Griewank(x, M, opt)
    elif task == 2:
        M = data["Rotation_Task2"]
        opt = data["GO_Task2"]
        l, u, d = -0.5, 0.5, 50
        x = l + array * (u - l)
        x = x[:d]
        return Weierstrass(x, M, opt)


def NILS(array, task, data):
    if task == 1:
        M = data["Rotation_Task1"]
        opt = data["GO_Task1"]
        l, u, d = -50, 50, 50
        x = l + array * (u - l)
        x = x[:d]
        return Rastrgin(x, M, opt)
    elif task == 2:
        l, u, d = -500, 500, 50
        x = l + array * (u - l)
        x = x[:d]
        return Schwefel(x)


# 读取数据
def get_data(question):
    # 完全相交
    if question == "CIHS":
        data = sio.loadmat("./Tasks/CI_H.mat")
    elif question == "CIMS":
        data = sio.loadmat("./Tasks/CI_M.mat")
    elif question == "CILS":
        data = sio.loadmat("./Tasks/CI_L.mat")
    # 部分相交
    elif question == "PIHS":
        data = sio.loadmat("./Tasks/PI_H.mat")
    elif question == "PIMS":
        data = sio.loadmat("./Tasks/PI_M.mat")
    elif question == "PILS":
        data = sio.loadmat("./Tasks/PI_L.mat")
    # 没有交集
    elif question == "NIHS":
        data = sio.loadmat("./Tasks/NI_H.mat")
    elif question == "NIMS":
        data = sio.loadmat("./Tasks/NI_M.mat")
    elif question == "NILS":
        data = sio.loadmat("./Tasks/NI_L.mat")
    return data


class Tasks():
    def __init__(self, question, task):
        self.question = question  # 问题
        self.task = task  # 任务
        self.data = get_data(self.question)  # 数据

    def function(self, array):
        # 完全相交
        if self.question == "CIHS":
            fitness = CIHS(array, self.task, self.data)
        elif self.question == "CIMS":
            fitness = CIMS(array, self.task, self.data)
        elif self.question == "CILS":
            fitness = CILS(array, self.task, self.data)
        # 部分相交
        elif self.question == "PIHS":
            fitness = PIHS(array, self.task, self.data)
        elif self.question == "PIMS":
            fitness = PIMS(array, self.task, self.data)
        elif self.question == "PILS":
            fitness = PILS(array, self.task, self.data)
        # 没有交集
        elif self.question == "NIHS":
            fitness = NIHS(array, self.task, self.data)
        elif self.question == "NIMS":
            fitness = NIMS(array, self.task, self.data)
        elif self.question == "NILS":
            fitness = NILS(array, self.task, self.data)
        return fitness
