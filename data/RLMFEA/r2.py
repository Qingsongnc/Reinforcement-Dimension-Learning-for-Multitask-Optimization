import algorithm
import pandas as pd
import numpy as np
import WCCI2020MTSO as WCCI
import CEC2017MTSO as CEC

# 此为算法运行
def run_2017(questions):
    path = './results/'
    # 遍历问题
    Means = []
    Stds = []
    Mins = []
    for question in questions:
        task1 = CEC.Tasks(question, 1)
        task2 = CEC.Tasks(question, 2)
        T1 = []
        T2 = []
        # 重复运行
        for i in range(res):
            t1, t2, t1_gbest_F, t2_gbest_F = algorithm.algorithm(task1, task2, question)
            T1.append(t1)
            T2.append(t2)
        T1 = np.array(T1)
        T2 = np.array(T2)
        # 求最后一列的平均值
        last_column_mean_task1 = np.mean(T1[:, -1])
        last_column_mean_task2 = np.mean(T2[:, -1])
        # 求最后一列的标准差
        last_column_std_task1 = np.std(T1[:, -1])
        last_column_std_task2 = np.std(T2[:, -1])
        # 求最后一列的最小值
        last_column_min_task1 = np.min(T1[:, -1])
        last_column_min_task2 = np.min(T2[:, -1])
        Means.append([last_column_mean_task1, last_column_mean_task2])
        Stds.append([last_column_std_task1, last_column_std_task2])
        Mins.append([last_column_min_task1, last_column_min_task2])
        df1 = pd.DataFrame(T1)
        df2 = pd.DataFrame(T2)
        # 将数据写入两个不同的Excel表格
        with pd.ExcelWriter(path + question + '.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Task1', index=False)
            df2.to_excel(writer, sheet_name='Task2', index=False)
    # 写入均值、标准差、最小值
    df1 = pd.DataFrame(Means)
    df2 = pd.DataFrame(Stds)
    df3 = pd.DataFrame(Mins)
    df1.index = questions
    df2.index = questions
    df3.index = questions
    with pd.ExcelWriter('./results/result-2017.xlsx') as writer:
        df1.to_excel(writer, sheet_name='means', index_label='question')
        df2.to_excel(writer, sheet_name='std', index_label='question')
        df3.to_excel(writer, sheet_name='min', index_label='question')


def run_2020(questions):
    path = './results/'
    # 遍历问题
    Means = []
    Stds = []
    Mins = []
    for question in questions:
        if question == 1:
            task1, task2 = WCCI.Benchmark1()
        elif question == 2:
            task1, task2 = WCCI.Benchmark2()
        elif question == 3:
            task1, task2 = WCCI.Benchmark3()
        elif question == 4:
            task1, task2 = WCCI.Benchmark4()
        elif question == 5:
            task1, task2 = WCCI.Benchmark5()
        elif question == 6:
            task1, task2 = WCCI.Benchmark6()
        elif question == 7:
            task1, task2 = WCCI.Benchmark7()
        elif question == 8:
            task1, task2 = WCCI.Benchmark8()
        elif question == 9:
            task1, task2 = WCCI.Benchmark9()
        elif question == 10:
            task1, task2 = WCCI.Benchmark10()
        T1 = []
        T2 = []
        # 重复运行
        for i in range(res):
            t1, t2, t1_gbest_F, t2_gbest_F = algorithm.algorithm(task1, task2, question)
            T1.append(t1)
            T2.append(t2)
        T1 = np.array(T1)
        T2 = np.array(T2)
        # 求最后一列的平均值
        last_column_mean_task1 = np.mean(T1[:, -1])
        last_column_mean_task2 = np.mean(T2[:, -1])
        # 求最后一列的标准差
        last_column_std_task1 = np.std(T1[:, -1])
        last_column_std_task2 = np.std(T2[:, -1])
        # 求最后一列的最小值
        last_column_min_task1 = np.min(T1[:, -1])
        last_column_min_task2 = np.min(T2[:, -1])
        Means.append([last_column_mean_task1, last_column_mean_task2])
        Stds.append([last_column_std_task1, last_column_std_task2])
        Mins.append([last_column_min_task1, last_column_min_task2])
        df1 = pd.DataFrame(T1)
        df2 = pd.DataFrame(T2)
        # 将数据写入两个不同的Excel表格
        with pd.ExcelWriter(path + 'Benchmark'+str(question) + '.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Task1', index=False)
            df2.to_excel(writer, sheet_name='Task2', index=False)
    # 写入均值、标准差、最小值
    df1 = pd.DataFrame(Means)
    df2 = pd.DataFrame(Stds)
    df3 = pd.DataFrame(Mins)
    df1.index = questions
    df2.index = questions
    df3.index = questions
    with pd.ExcelWriter('./results/result-2020.xlsx') as writer:
        df1.to_excel(writer, sheet_name='means', index_label='question')
        df2.to_excel(writer, sheet_name='std', index_label='question')
        df3.to_excel(writer, sheet_name='min', index_label='question')


# 基本问题
questions1 = ["CIHS", "CIMS", "CILS", "PIHS", "PIMS", "PILS", "NIHS", "NIMS", "NILS"]
questions2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
res =30 # 重复运行次数


run_2020(questions2)