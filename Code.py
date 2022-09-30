import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import threading
import math
import csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from sklearn import discriminant_analysis
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pickle


# 轮盘赌选择
def selection(population, fitness, n):
    '''
    :param population: 种群
    :param fitness: 每一个个体的适应度函数
    :param n: 初始种群的个体数
    '''
    fitness_sum = np.zeros(n)
    for i in range(n):
        if i == 0:
            fitness_sum[i] = fitness[i]
        else:
            fitness_sum[i] = fitness[i] + fitness_sum[i - 1]
    for i in range(n):
        fitness_sum[i] = fitness_sum[i] / sum(fitness)

    # 选择新的种群
    population_new = np.zeros((n, feature_num))
    for i in range(n):
        rand = np.random.uniform(0, 1)
        for j in range(n):
            if j == 0:
                if rand <= fitness_sum[j]:
                    population_new[i] = population[j]
            else:
                if fitness_sum[j - 1] < rand and rand <= fitness_sum[j]:
                    population_new[i] = population[j]
    return population_new


# 二父辈交叉编译函数
def job(i):
    np.random.seed(0)
    rand = random.random()
    cross_sit = random.random()
    # % rand < 交叉概率，对两个个体的染色体串进行交叉操作
    if (rand < cross_rate):
        cross_position = int(feature_num * cross_sit)
        temp = population_2[i, cross_position: feature_num]
        population_2[i, cross_position: feature_num] = population_2[i + 1, cross_position: feature_num]
        population_2[i + 1, cross_position: feature_num] = temp

def crossover_two():
    np.random.shuffle(population_2)
    l = len(population_2)
    threads = []
    for i in range(0, l, 2):
        threads.append(threading.Thread(target=job, args=(i,)))
    for t in threads:
        t.setDaemon(True)  # 声明t为守护线程，设置的话，子线程将和主线程一起运行，并且直接结束，不会再执行循环里面的子线程
        t.start()
    t.join()


# 三父辈交叉编译函数
def crossover_three2(fitness, Index):
    l = len(population)
    population1 = []
    population2 = []
    population3 = []
    fitsum1 = 0
    fitsum2 = 0
    fitsum3 = 0
    for i in range(math.floor(l / 3)): # 取适应度值最高的三分之一部分
        population1.append(population[Index[i]])
        fitsum1 += fitness[Index[i]]
        fitsum1 = fitsum1 / math.floor(l / 3)
    for i in range(math.floor(l / 3), math.floor(2 * l / 3)): # 取适应度值次高的三分之一部分
        population2.append(population[Index[i]])
        fitsum2 += fitness[Index[i]]
        fitsum2 = fitsum2 / math.floor(l / 3)
    for i in range(math.floor(2 * l / 3), l): # 取适应度值最低的三分之一部分
        population3.append(population[Index[i]])
        fitsum3 += fitness[Index[i]]
        fitsum3 = fitsum3 / (l - math.floor(l / 3) - math.floor(l / 3))

    population1 = np.array(population1)
    population2 = np.array(population2)
    population3 = np.array(population3)
    totalsum = fitsum1 + fitsum2 + fitsum3
    #计算选择的概率值
    ratio1 = fitsum1 / totalsum
    ratio2 = fitsum2 / totalsum
    np.random.seed(0)
    tem_number = 0
    for k in range(0, l - 1, 3):
        cross = random.random()
        if cross < cross_rate:
            select_num=[]
            select_index=[]
            combine=[]
            for i in range(3): #对于每一次选择的父辈个体，确定选择的编号
                rand = random.random()
                if rand <= ratio1:
                    ra1=random.randint(0, len(population1) - 1)
                    select_num.append(ra1)
                    select_index.append(1)
                    ll1 = population1[ra1]
                    combine.append(ll1[0:math.floor(len(population1[ra1]) / 3)])
                    combine.append(ll1[math.floor(len(population1[ra1]) / 3):math.floor(2 * len(population1[ra1]) / 3)])
                    combine.append(ll1[math.floor(2 * len(population1[ra1]) / 3):])
                elif rand <= ratio1 + ratio2:
                    ra2=random.randint(0, len(population2) - 1)
                    select_num.append(ra2)
                    select_index.append(2)
                    combine.append(population2[ra2][0:math.floor(len(population2[ra2]) / 3)])
                    combine.append(
                        population2[ra2][
                        math.floor(len(population2[ra2]) / 3): math.floor(2 * len(population2[ra2]) / 3)])
                    combine.append(population2[ra2][math.floor(2 * len(population2[ra2]) / 3):])
                else:
                    ra3=random.randint(0, len(population3) - 1)
                    select_num.append(ra3)
                    select_index.append(3)
                    combine.append(population3[ra3][0:math.floor(len(population3[ra3]) / 3)])
                    combine.append(
                        population3[ra3][
                        math.floor(len(population3[ra3]) / 3): math.floor(2 * len(population3[ra3]) / 3)])
                    combine.append(population3[ra3][math.floor(2 * len(population3[ra3]) / 3):])

            fitresult = []
            add = np.zeros((int(len(population[0]) / 3),), dtype=np.int)
            add2 = np.zeros((int(len(population[0]) / 3)+1,), dtype=np.int)
            add = np.array(add)
            add = add.tolist()
            add2 = np.array(add2)
            add2 = add2.tolist()

            temp_com = []

            for i in range(len(combine)):
                if i%3 == 1:
                    temp_combine = []
                    temp_combine.extend(combine[i])
                    temp_combine.extend(add)
                    temp_combine.extend(add)
                    #temp_combine.extend(add2)
                    temp_combine = np.array(temp_combine)
                    temp_com.append(temp_combine)
                    fitresult.append(Jd2(temp_combine, popu_orig, feature_num)) #计算每一份的适应度值
                elif i%3 == 2:
                    temp_combine = []
                    temp_combine.extend(add)
                    temp_combine.extend(combine[i])
                    temp_combine.extend(add)
                    temp_combine = np.array(temp_combine)
                    temp_com.append(temp_combine)
                    fitresult.append(Jd2(temp_combine, popu_orig, feature_num))  # 计算每一份的适应度值
                else:
                    temp_combine = []
                    temp_combine.extend(add)
                    #temp_combine.extend(add2)
                    temp_combine.extend(add)
                    temp_combine.extend(combine[i])
                    temp_combine = np.array(temp_combine)
                    temp_com.append(temp_combine)
                    fitresult.append(Jd2(temp_combine, popu_orig, feature_num))  # 计算每一份的适应度值

            fitresult = np.array(fitresult)
            Index_fit = np.argsort(-fitresult)

            llen = len(population)
            for i in range(3):
                if fitresult[Index_fit[i]] > 0.7:
                    population[tem_number + i] = temp_com[Index_fit[i]]
                else:
                    population[tem_number + i] = population[tem_number + i]

            tem_number += 3

# 三父辈变异操作
def job3(i):
    np.random.seed(0)
    c = np.random.uniform(0, 1)
    if c <= mutation_rate:  # pc变异概率
        mutation_s = population[i]
        zero = []  # zero存的是变异个体中第几个数为0
        one = []  # one存的是变异个体中第几个数为1
        for j in range(feature_num):
            if mutation_s[j] == 0:
                zero.append(j)
            else:
                one.append(j)
        if (len(zero) != 0) and (len(one) != 0):
            a = np.random.randint(0, len(zero))  # e是随机选择由0变为1的位置
            b = np.random.randint(0, len(one))  # f是随机选择由1变为0的位置
            e = zero[a]
            f = one[b]
            mutation_s[e] = 1
            mutation_s[f] = 0
            population[i] = mutation_s

def mutation_3():
    n = len(population)
    np.random.seed(0)
    threads = []
    for i in range(n):
        threads.append(threading.Thread(target=job3, args=(i,)))
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()


# 二父辈变异操作
def job2(i):
    np.random.seed(0)
    c = np.random.uniform(0, 1)
    if c <= mutation_rate:  # pc变异概率
        mutation_s = population_2[i]
        zero = []  # zero存的是变异个体中第几个数为0
        one = []  # one存的是变异个体中第几个数为1
        for j in range(feature_num):
            if mutation_s[j] == 0:
                zero.append(j)
            else:
                one.append(j)
        if (len(zero) != 0) and (len(one) != 0):
            a = np.random.randint(0, len(zero))  # e是随机选择由0变为1的位置
            b = np.random.randint(0, len(one))  # f是随机选择由1变为0的位置
            e = zero[a]
            f = one[b]
            mutation_s[e] = 1
            mutation_s[f] = 0
            population_2[i] = mutation_s

def mutation_2():
    n = len(population_2)
    np.random.seed(0)
    threads = []
    for i in range(n):
        threads.append(threading.Thread(target=job2, args=(i,)))
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()


# 个体适应度函数 Jd(x)，x是d维特征向量(1*60维的行向量,1表示选择该特征)
def Jd(x, sonar2, feature_num):
    d = len(x)  # 从特征向量x中提取出相应的特征
    Feature = np.zeros(d)  # 数组Feature用来存x选择的是哪d个特征
    k = 0
    for i in range(feature_num):
        if x[i] == 1:
            Feature[k] = i
            k += 1

    if k == 0:
        return 0

    sonar2 = np.array(sonar2)  # 将d个特征从sonar2数据集中取出重组成一个208*d的矩阵sonar3
    sonar3 = np.zeros((total_sample, 1))
    for i in range(len(Feature)):
        p = Feature[i]
        p = p.astype(int)
        q = sonar2[:, p]
        q = q.reshape(total_sample, 1)
        sonar3 = np.append(sonar3, q, axis=1)
    sonar3 = np.delete(sonar3, 0, axis=1)
    sonar2 = sonar2.astype(float)
    sonar3 = sonar3.astype(float)

    # 求类间离散度矩阵Sb
    sonar3_1 = sonar3[0:pos_sample, :]  # sonar数据集分为两类
    sonar3_2 = sonar3[pos_sample:total_sample, :]
    m = np.mean(sonar3, axis=0)  # 总体均值向量
    m1 = np.mean(sonar3_1, axis=0)  # 第一类的均值向量
    m2 = np.mean(sonar3_2, axis=0)  # 第二类的均值向量
    m = m.reshape(d, 1)  # 将均值向量转换为列向量以便于计算
    m1 = m1.reshape(d, 1)
    m2 = m2.reshape(d, 1)
    Sb = ((m1 - m).dot((m1 - m).T) * (pos_sample / total_sample) + (m2 - m).dot((m2 - m).T) * (
                (total_sample - pos_sample) / total_sample))  # 除以类别个数

    # 求类内离散度矩阵Sw
    S1 = np.zeros((d, d))
    S2 = np.zeros((d, d))
    for i in range(pos_sample):
        S1 += (sonar3_1[i].reshape(d, 1) - m1).dot((sonar3_1[i].reshape(d, 1) - m1).T)
    S1 = S1 / pos_sample
    for i in range((total_sample - pos_sample)):
        S2 += (sonar3_2[i].reshape(d, 1) - m2).dot((sonar3_2[i].reshape(d, 1) - m2).T)
    S2 = S2 / (total_sample - pos_sample)

    Sw = (S1 * (pos_sample / total_sample) + S2 * ((total_sample - pos_sample) / total_sample))
    # 计算个体适应度函数 Jd(x)
    J1 = np.trace(Sb)
    J2 = np.trace(Sw)
    Jd = J1 / J2

    return Jd


def Jd2(xx, sonar2, feature_num):
    Feature = []  # 数组Feature用来存x选择的是哪d个特征
    k = 0
    for i in range(feature_num):
        if xx[i] == 1:
            Feature.append(i)
            k += 1

    if k == 0:
        return 0

    x1 = []
    for i in range(total_sample):
        l1 = len(Feature)
        temp1 = []
        for j in range(l1):
            temp1.append(x[i][Feature[j]])
        x1.append(temp1)
    # 用分类器进行评测，查看选择出来的特征的分类效果
    x1 = np.array(x1, dtype='float_')
    yy = np.array(y)
    X1_train, X1_test, y1_train, y1_test = train_test_split(x1, yy, test_size=0.3, random_state=1)

    #clf1 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf1 = XGBClassifier()

    return clf1.fit(X1_train, y1_train).score(X1_test, y1_test)


def binary_code(initial_population_num, feature_selection_num):
    '''
    :return: 对种群的个体进行二进制随机编码，控制最初的1 的个数为被选择的特征数 feature_num
    '''
    population = np.zeros((initial_population_num, feature_num))  # 初始化种群
    np.random.seed(0)
    for i in range(initial_population_num):  # 定义种群的个体数为n
        a = np.zeros(feature_num - feature_selection_num)
        b = np.ones(feature_selection_num)  # 将选择的d维特征定义为个体c中的1
        c = np.append(a, b)
        c = (np.random.permutation(c.T)).T  # 随机生成一个d维的个体
        population[i] = c  # 初代的种群为 population，共有n个个体
    return population




global feature_num  # 特征数
global cross_rate  # 交叉概率
global mutation_rate  # 变异概率
global total_sample  # 总样本数
global pos_sample  # 正样本数
global population  # 三父辈模拟种群
global population_2  # 二父辈模拟种群
global iteration_time  # 迭代次数

feature_num = 60
cross_rate = 0.8
mutation_rate = 0.3
total_sample = 208
pos_sample = 97
# pos_sample = 207
iteration_time = 30
pop_select = 208

# population = binary_code(total_sample, 10)

column = []
for i in range(feature_num):
    with open("/Users/songhaoqiu/Desktop/Graduation project/sonar.all-data.csv") as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[i] for row in reader]
        column1 = list(map(float, column1))
        column.append(column1)

with open("/Users/songhaoqiu/Desktop/Graduation project/sonar.all-data.csv") as csvfile:
    reader = csv.reader(csvfile)
    lb = [row[feature_num] for row in reader]
lb_ = []
for i in range(len(lb)):
    if(lb[i]=='R'):
        lb_.append(1)
    else:
        lb_.append(-1)

R = []
Count = 0
for i in range(len(column)):
    column[i]=np.array(column[i],dtype = 'float_')
    lb_=np.array(lb_,dtype = 'float_')
    r, p=stats.pearsonr(column[i], lb_)
    R.append(r)
R = np.array(R)
R_index = np.argsort(-R)
R_index = list(R_index)
R_index = R_index[0:50]

population = np.zeros((total_sample, feature_num))
for i in range(len(population)):
    ppp = random.sample(R_index, 10)
    for j in range(len(ppp)):
        population[i][ppp[j]]=1

population_2 = population.copy()

# 读取数据文件
with open("/Users/songhaoqiu/Desktop/Graduation project/sonar.all-data.csv") as csvfile:
    reader = csv.reader(csvfile)
    data = []
    for row in reader:
        data.append(row)
x = []
y = []
scale = len(data)
for i in range(scale):
    t = []
    a = len(data[i])
    for j in range(a - 1):
        t.append(data[i][j])
    x.append(t)
    y.append(data[i][a - 1])

x=np.array(x,dtype = 'float_')
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

sonar = pd.read_csv('/Users/songhaoqiu/Desktop/Graduation project/sonar.all-data.csv', header=None, sep=',')
xx = sonar.iloc[0:total_sample, 0:feature_num]
popu_orig = np.mat(xx)

min_max_scaler = preprocessing.MinMaxScaler()
popu_orig = min_max_scaler.fit_transform(popu_orig)

"""
feature_num = 166
cross_rate = 0.8
mutation_rate = 0.3
total_sample = 476
pos_sample = 207
iteration_time = 10
pop_select = 200

column = []
for i in range(feature_num):
    with open("/Users/songhaoqiu/Desktop/Graduation project/clean1.csv") as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[i] for row in reader]
        column1 = list(map(float, column1))
        column.append(column1)

with open("/Users/songhaoqiu/Desktop/Graduation project/clean1.csv") as csvfile:
    reader = csv.reader(csvfile)
    lb = [row[feature_num] for row in reader]
lb_ = []
for i in range(len(lb)):
    if(lb[i]=='1'):
        lb_.append(1)
    else:
        lb_.append(-1)

R = []
Count = 0
for i in range(len(column)):
    column[i]=np.array(column[i],dtype = 'float_')
    lb_=np.array(lb_,dtype = 'float_')
    r, p=stats.pearsonr(column[i], lb_)
    R.append(r)
R = np.array(R)
R_index = np.argsort(-R)
R_index = list(R_index)
R_index = R_index[0:160]

population = np.zeros((total_sample, feature_num))
for i in range(len(population)):
    ppp = random.sample(R_index, 30)
    for j in range(len(ppp)):
        population[i][ppp[j]]=1

population_2 = population.copy()

# 读取数据文件
with open("/Users/songhaoqiu/Desktop/Graduation project/clean1.csv") as csvfile:
    reader = csv.reader(csvfile)
    data = []
    for row in reader:
        data.append(row)
x = []
y = []
scale = len(data)
for i in range(scale):
    t = []
    a = len(data[i])
    for j in range(a - 1):
        t.append(data[i][j])
    x.append(t)
    y.append(data[i][a - 1])

x=np.array(x,dtype = 'float_')
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

sonar = pd.read_csv('/Users/songhaoqiu/Desktop/Graduation project/clean1.csv', header=None, sep=',')
xx = sonar.iloc[0:total_sample, 0:feature_num]
popu_orig = np.mat(xx)

min_max_scaler = preprocessing.MinMaxScaler()
popu_orig = min_max_scaler.fit_transform(popu_orig)
"""
def function(population, population_2,y):

    count = 1
    temp = 0
    temp_2 = 0
    plot = []
    plot_2 = []
    b1 = 0
    b2 = 0
    plot.append(0)
    plot_2.append(0)

    while (count <= iteration_time):
        print(count)
        fitness = []
        fitness_2 = []
        for i in range(pop_select):  # 对于种群中的每一个个体计算其适应度值
            fitness.append(Jd(population[i], popu_orig, feature_num))
            fitness_2.append(Jd(population_2[i], popu_orig, feature_num))

        if count != 1:  # 存储每一次的迭代后所得到的最大的适应度值
            plot.append(abs(max(fitness)))
            plot_2.append(abs(max(fitness_2)))

        # 确定种群当中适应度值最大的个体的位置（下标）
        b1 = fitness.index(max(fitness))
        b2 = fitness_2.index(max(fitness_2))

        fit=np.array(fitness)
        fit2=np.array(fitness_2)
        Index = np.argsort(-fit)
        Index2 = np.argsort(-fit2)
        save=[]
        save_2=[]

        # 将适应度值最大的5个个体的下标及其本身进行保存
        for i in range(5):
            save.append(Index[i])
            save_2.append(Index2[i])
        population_save=[]
        population_save2=[]
        for h in range(5):
            population_save.append(population[save[h]])
            population_save2.append(population_2[save_2[h]])

        # 二、三父辈的种群进行选择操作
        population = selection(population, fitness, pop_select)
        population_2 = population.copy()

        # 三父辈进行一次交叉变异
        crossover_three2(fitness,Index)
        mutation_3()

        # 二父辈进行一次交叉变异
        crossover_two()
        mutation_2()

        # 计算交叉变异后的新的种群个体的适应度值
        fitness_new = []
        fitness_new2 = []
        for i in range(pop_select):
            fitness_new.append(Jd(population[i], popu_orig, feature_num))
            fitness_new2.append(Jd(population_2[i], popu_orig, feature_num))

        fit_new = np.array(fitness_new)
        fit_new2 = np.array(fitness_new2)
        Index_new = np.argsort(-fit_new)
        Index2_new = np.argsort(-fit_new2)

        # 将三父辈中的新得到的种群当中的适应度最差的五个个体换掉
        l = len(population)
        population_temp=population.copy()
        for i in range(l):
            population_temp[i]=population[Index_new[i]]  # 将种群按适应度值重新排序
        population = population_temp[0:l - 5]  # 去掉当前最差的五个个体
        population = population.tolist()
        for i in range(5):
            population.append(population_save[i])  # 增加之前所得到的最好的五个个体
        population=np.array(population)

        # 将二父辈中的新得到的种群当中的适应度最差的十个个体换掉
        l2 = len(population_2)
        population_temp2 = population_2.copy()
        for i in range(l):
            population_temp2[i] = population_2[Index2_new[i]]
        population_2 = population_temp2[0:l2 - 5]
        population_2 = population_2.tolist()
        for i in range(5):
            population_2.append(population_save2[i])
        population_2 = np.array(population_2)

        count += 1

    xxx = [i for i in range(0, iteration_time)]
    plt.plot(xxx, plot, mec='r', mfc='w', label='Three')
    plt.plot(xxx, plot_2, ms=10, label='Two')
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.ylim(0, 0.2)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('Iteration_times')  # X轴标签
    plt.ylabel("Convergence")  # Y轴标签
    plt.title("Comparison")  # 标题
    plt.show()

    max_popu = population[b1]      # 通过三父辈迭代后得到的种群中适应度值最好的个体
    max_popu_2 = population_2[b2]  # 通过二父辈迭代后得到的种群中适应度值最好的个体

    print(population[b1])
    print(population_2[b2])

    # 特征选择，将适应度值最好个体当中的所选特征作为我们最终所选择的特征
    fea_select = []
    fea_select_2 = []
    for i in range(feature_num):
        if max_popu[i] == 1:
            fea_select.append(i)
        if max_popu_2[i] == 1:
            fea_select_2.append(i)

    # 将确定好要选择的特征重新存入新序列当中，作为分类所使用的个体
    x1 = []
    x2 = []
    for i in range(total_sample):
        l1 = len(fea_select)
        l2 = len(fea_select_2)
        temp1 = []
        temp2 = []
        for j in range(l1):
            temp1.append(x[i][fea_select[j]])
        for j in range(l2):
            temp2.append(x[i][fea_select_2[j]])
        x1.append(temp1)
        x2.append(temp2)

    # 用分类器进行评测，查看选择出来的特征的分类效果
    x1 = np.array(x1, dtype = 'float_')
    x2 = np.array(x2, dtype = 'float_')
    y = np.array(y)
    X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.3, random_state=1)
    X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.3, random_state=1)

    #clf1 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    #clf2 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')

    clf1 = XGBClassifier()
    clf2 = XGBClassifier()

    result1 = clf1.fit(X1_train, y1_train).score(X1_test, y1_test)
    result2 = clf2.fit(X2_train, y2_train).score(X2_test, y2_test)
    print(result1)
    print(result2)
    return 0


print("Input:")
intro=input()
if(intro=="start"):
    print(function(population, population_2, y))
else:
    print("Thank you for using this program! Again input start to find the result!")