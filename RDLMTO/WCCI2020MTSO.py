import numpy as np
import os
#此为WCCI2020MTSO测试集，使用方法为：
# task1,task2=Benchmark10()
# task1.function(array)

# 任务父类
class Task:
    def __init__(self, Low=0, High=1, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        self.Low = Low
        self.High = High
        self.Dimension = n
        if bias is not None:
            self.center = bias
        else:
            self.center = np.zeros(shape=n)
        if coeffi is not None:
            self.M = coeffi
        else:
            self.M = np.zeros(shape=(n, n))
            for i in range(n):
                self.M[i, i] = 1
        self.shuffle=shuffle
        self.sh_rate=sh_rate

    def decode(self, X):
        X1 = self.Low + (self.High - self.Low) * X
        X1=X1[0:self.Dimension]
        X1 = X1 - self.center
        X1=X1*self.sh_rate
        return np.dot(self.M, X1.T)




# 函数子类

class Ellips(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)

    def function(self, X,tag=False):
        if tag==False:
            temp = self.decode(X)
        else:
            temp=X
        indexs = np.arange(self.Dimension)
        return np.sum((10 ** (6 * indexs / (self.Dimension - 1))) * temp * temp)

    def Info(self):
        return 'Ellips ' + str(self.Dimension)


class Discus(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)

    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)
        else:
            temp = X
        return temp[0] * temp[0] * (10 ** 6) + np.sum(temp[1:] * temp[1:])

    def Info(self):
        return 'Discus ' + str(self.Dimension)


class Rosenbrock(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=2.048 / 100.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)


    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)
        else:
            temp = X* self.sh_rate
        return np.sum(100 * ((temp[:self.Dimension - 1] ** 2 - temp[1:]) ** 2) + (temp[:self.Dimension - 1] - 1) ** 2)

    def Info(self):
        return 'Rosenbrock ' + str(self.Dimension)


class Ackley(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)

    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)
        else:
            temp = X
        return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.sum(temp ** 2) / self.Dimension)) - np.exp(
            np.sum(np.cos(2 * np.pi * temp)) / self.Dimension) + 500

    def Info(self):
        return 'Ackley ' + str(self.Dimension)


class Weierstrass(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=0.5/100):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)


    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)
        else:
            temp = X* self.sh_rate
        a = 0.5
        b = 3
        kmax = 20
        sums = 0
        for k in range(0, kmax + 1):
            sums += np.sum(a ** k * np.cos(2 * np.pi * b ** k * (temp + 0.5))) - self.Dimension * a ** k * np.cos(
                np.pi * b ** k)
        return sums + 600

    def Info(self):
        return 'Weierstrass ' + str(self.Dimension)


class Griewank(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=600.0/100):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)


    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)
        else:
            temp = X* self.sh_rate
        return 1 + np.sum(temp ** 2) / 4000 - np.prod(np.cos(temp / np.sqrt(np.arange(1, self.Dimension + 1)))) + 700

    def Info(self):
        return 'Griewank ' + str(self.Dimension)


class Rastrigin(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=5.12/100):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)


    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)
        else:
            temp = X* self.sh_rate
        return np.sum(temp ** 2 - 10 * np.cos(2 * np.pi * temp) + 10)

    def Info(self):
        return 'Rastrigin ' + str(self.Dimension)


class Schwefel(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1000.0/100):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)

    def function(self, X,tag=False):
        if tag==False:
            z = self.decode(X)+ 4.209687462275036e+002
        else:
            temp = X * self.sh_rate
            z=temp+ 4.209687462275036e+002
        tmp=z.copy()
        z[z < -500] = -500 + np.fmod(np.abs(z[z < -500]), 500)
        z[z > 500] = 500 - np.fmod(np.abs(z[z > 500]), 500)
        return 4.189828872724338e+002 * self.Dimension - np.sum(z * np.sin(np.sqrt(np.abs(z)))) + np.sum(
            (tmp[tmp < -500] + 500) ** 2 / 10000 / self.Dimension) + np.sum(
            (tmp[tmp > 500] - 500) ** 2 / 10000 / self.Dimension) + 1100

    def Info(self):
        return 'Schwefel ' + str(self.Dimension)


class Katsuura(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=5.0/100.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)


    def function(self, X,tag=False):
        if tag == False:
            z = self.decode(X)
        else:
            z = X * self.sh_rate
        nx=self.Dimension
        f = 1.0
        tmp3 = np.power(1.0 * nx, 1.2)
        for i in range(nx):
            temp = 0.0
            for j in range(1, 33):
                tmp1 = np.power(2.0, j)
                tmp2 = tmp1 * z[i]
                temp += np.abs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
            f *= np.power(1.0 + (i + 1) * temp, 10.0 / tmp3)
        tmp1 = 10.0 / nx / nx
        f = f * tmp1 - tmp1
        return f

    def Info(self):
        return 'Katsuura ' + str(self.Dimension)


class GrieRosen(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=5.0/100.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)


    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)+1
        else:
            temp = X * self.sh_rate+1
        temp1 = np.append(temp[1:], temp[0])
        temp1 = 100 * (temp * temp - temp1) * (temp * temp - temp1) + (temp - 1) * (temp - 1)
        return np.sum(temp1 * temp1 / 4000 - np.cos(temp1) + 1) + 1500

    def Info(self):
        return 'Grie_Rosen ' + str(self.Dimension)


class Escaffer6(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)

    def function(self, X,tag=False):
        if tag == False:
            z = self.decode(X)
        else:
            z = X
        nx=self.Dimension
        f=0.0
        for i in range(nx - 1):
            temp1 = np.sin(np.sqrt(z[i] ** 2 + z[i + 1] ** 2))
            temp1 = temp1 ** 2
            temp2 = 1.0 + 0.001 * (z[i] ** 2 + z[i + 1] ** 2)
            f += 0.5 + (temp1 - 0.5) / (temp2 ** 2)
        temp1 = np.sin(np.sqrt(z[nx - 1] ** 2 + z[0] ** 2))
        temp1 = temp1 ** 2
        temp2 = 1.0 + 0.001 * (z[nx - 1] ** 2 + z[0] ** 2)
        f += 0.5 + (temp1 - 0.5) / (temp2 ** 2)
        return f+1600

    def Info(self):
        return 'Escaffer6 ' + str(self.Dimension)


class HappyCat(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=5.0/100.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)

    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)- 1
        else:
            temp = X * self.sh_rate-1
        r2=np.sum(temp * temp)
        sum_z=np.sum(temp)
        return np.abs(r2-self.Dimension)**(1/4)+(0.5*r2+sum_z)/self.Dimension+0.5+1300


    def Info(self):
        return 'Happycat ' + str(self.Dimension)


class Hgbat(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=5.0/100.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)

    def function(self, X,tag=False):
        if tag == False:
            temp = self.decode(X)- 1
        else:
            temp = X * self.sh_rate-1
        return np.sqrt(np.abs((np.sum(temp * temp) ** 2 - np.sum(temp) ** 2))) + (
                np.sum(temp * temp) / 2 + np.sum(temp)) / self.Dimension + 0.5

    def Info(self):
        return 'Hgbat ' + str(self.Dimension)


class Hf01(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)
        self.Sch = Schwefel(n=15)
        self.Ras = Rastrigin(n=15)
        self.Elp = Ellips(n=20)

    def function(self, X):
        x = self.decode(X)
        nx=self.Dimension
        temp = np.zeros(nx)
        for i in range(nx):
            temp[i]=x[int(self.shuffle[i])-1]
        func = 0
        func += self.Sch.function(temp[:15],tag=True)
        func += self.Ras.function(temp[15:30],tag=True)
        func += self.Elp.function(temp[30:],tag=True)
        return func + 600

    def Info(self):
        return 'Hf01 ' + str(self.Dimension)


class Hf04(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)
        self.Hg = Hgbat(n=10)
        self.Dis = Discus(n=10)
        self.GR = GrieRosen(n=15)
        self.Ras = Rastrigin(n=15)

    def function(self, X):
        x = self.decode(X)
        nx = self.Dimension
        temp = np.zeros(nx)
        for i in range(nx):
            temp[i] = x[int(self.shuffle[i]) - 1]
        func = 0
        func += self.Hg.function(temp[:10],tag=True)
        func += self.Dis.function(temp[10:20],tag=True)
        func += self.GR.function(temp[20:35],tag=True)
        func += self.Ras.function(temp[35:],tag=True)
        return func + 500

    def Info(self):
        return 'Hf04 ' + str(self.Dimension)


class Hf05(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)
        self.Es = Escaffer6(n=5)
        self.Hg = Hgbat(n=10)
        self.Ros = Rosenbrock(n=10)
        self.Sch = Schwefel(n=10)
        self.Elp = Ellips(n=15)

    def function(self, X):
        x = self.decode(X)
        nx = self.Dimension
        temp = np.zeros(nx)
        for i in range(nx):
            temp[i] = x[int(self.shuffle[i]) - 1]
        func = 0
        func += self.Es.function(temp[:5],tag=True)
        func += self.Hg.function(temp[5:15],tag=True)
        func += self.Ros.function(temp[15:25],tag=True)
        func += self.Sch.function(temp[25:35],tag=True)
        func += self.Elp.function(temp[35:],tag=True)
        return func - 600

    def Info(self):
        return 'Hf05 ' + str(self.Dimension)


class Hf06(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,shuffle=None,sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias,shuffle,sh_rate)
        self.Kat = Katsuura(n=5)
        self.HC = HappyCat(n=10)
        self.GR = GrieRosen(n=10)
        self.Sch = Schwefel(n=10)
        self.Ack = Ackley(n=15)

    def function(self, X):
        x = self.decode(X)
        nx = self.Dimension
        temp = np.zeros(nx)
        for i in range(nx):
            temp[i] = x[int(self.shuffle[i]) - 1]
        func = 0
        func += self.Kat.function(temp[:5],tag=True)
        func += self.HC.function(temp[5:15],tag=True)
        func += self.GR.function(temp[15:25],tag=True)
        func += self.Sch.function(temp[25:35],tag=True)
        func += self.Ack.function(temp[35:],tag=True)
        return func - 2200

    def Info(self):
        return 'Hf06 ' + str(self.Dimension)


# 读取文件

def GetMatrixs(filepath):
    bias1 = np.loadtxt(filepath + 'bias_1')
    bias2 = np.loadtxt(filepath + 'bias_2')
    matrix1 = np.loadtxt(filepath + 'matrix_1')
    matrix2 = np.loadtxt(filepath + 'matrix_2')
    return {'bias1': bias1, 'bias2': bias2, 'ma1': matrix1, 'ma2': matrix2}

def GetShuffle(filepath):
    shuffle= np.loadtxt(filepath)
    return {'shuffle':shuffle}

# 10个基本问题

def Benchmark1(filename='./Tasks/benchmark_1/'):
    params = GetMatrixs(filename)
    Task1 = Weierstrass(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Weierstrass(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark2(filename='./Tasks/benchmark_2/'):
    params = GetMatrixs(filename)
    Task1 = Griewank(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Griewank(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark3(filename='./Tasks/benchmark_3/'):
    params = GetMatrixs(filename)
    s=GetShuffle('./Tasks/shuffle/shuffle_data_17_D50.txt')
    Task1 = Hf01(coeffi=params['ma1'], bias=params['bias1'],shuffle=s['shuffle'])
    Task2 = Hf01(coeffi=params['ma2'], bias=params['bias2'],shuffle=s['shuffle'])
    return [Task1, Task2]


def Benchmark4(filename='./Tasks/benchmark_4/'):
    params = GetMatrixs(filename)
    Task1 = HappyCat(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = HappyCat(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark5(filename='./Tasks/benchmark_5/'):
    params = GetMatrixs(filename)
    Task1 = GrieRosen(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = GrieRosen(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark6(filename='./Tasks/benchmark_6/'):
    params = GetMatrixs(filename)
    s = GetShuffle('./Tasks/shuffle/shuffle_data_21_D50.txt')
    Task1 = Hf05(coeffi=params['ma1'], bias=params['bias1'],shuffle=s['shuffle'])
    Task2 = Hf05(coeffi=params['ma2'], bias=params['bias2'],shuffle=s['shuffle'])
    return [Task1, Task2]


def Benchmark7(filename='./Tasks/benchmark_7/'):
    params = GetMatrixs(filename)
    s = GetShuffle('./Tasks/shuffle/shuffle_data_22_D50.txt')
    Task1 = Hf06(coeffi=params['ma1'], bias=params['bias1'],shuffle=s['shuffle'])
    Task2 = Hf06(coeffi=params['ma2'], bias=params['bias2'],shuffle=s['shuffle'])
    return [Task1, Task2]


def Benchmark8(filename='./Tasks/benchmark_8/'):
    params = GetMatrixs(filename)
    Task1 = Ackley(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Ackley(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark9(filename='./Tasks/benchmark_9/'):
    params = GetMatrixs(filename)
    Task1 = Schwefel(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Escaffer6(coeffi=params['ma2'], bias=params['bias2'])
    return [Task1, Task2]


def Benchmark10(filename='./Tasks/benchmark_10/'):
    params = GetMatrixs(filename)
    s = GetShuffle('./Tasks/shuffle/shuffle_data_20_D50.txt')
    Task1 = Hf04(coeffi=params['ma1'], bias=params['bias1'],shuffle=s['shuffle'])
    s = GetShuffle('./Tasks/shuffle/shuffle_data_21_D50.txt')
    Task2 = Hf05(coeffi=params['ma2'], bias=params['bias2'],shuffle=s['shuffle'])
    return [Task1, Task2]


