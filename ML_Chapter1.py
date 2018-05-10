# -*- coding: utf-8 -*-

"""
This is a temporary script file.


for i in range(2):
    val = input("请输入带有温度符号的温度值（例如31C）:")
    if val[-1] in['c', 'C']:
        f = 1.8 * float(val[0:-1]) + 32
        print("转化后的温度为:%.3fF" % f)
        # print("转化后的温度为:" + f)       # "+" 两端链接的是字符串
    elif val[-1] in ['F', 'f']:
        c = (float(val[0:-1]) - 32)/1.8
        print("转化后的温度为：%.2fC" % c)
        # print("转化后的温度为：" + c)
    else:
        print("输入有误！")


import random
def make_score(num):
    score = [random.ranint(0,100) for i in range(num)]
    return score
"""

# import numpy as np
# import pylab as pl
#
# x = np.random.uniform(1, 100, 1000)
# y = np.log(x) + np.random.normal(0, .3, 1000)
# pl.scatter(x, y, s=1, label="log(x) with noise")
# pl.plot(np.arange(1, 100), np.log(np.arange(1, 100)), c='b', label='log(x) true function')
# pl.xlabel("x")
# pl.ylabel("f(x) = log(x)")
# pl.legend(loc="best")
# pl.title("A basic Log Function")
# pl.show()


import pandas as pd
import os
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

os.chdir("E:/MyStudy/Python/PyScript/Python机器学习及实践/Datasets/Breast_Cancer")
df_train = pd.read_csv("breast-cancer-train.csv")
df_test = pd.read_csv("breast-cancer-test.csv")
print('-' * 35)
print(df_test[1:10])

# 构建测试集中的正负样本
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# plt.show()

intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0]) / coef[1]

lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])

predict = lr.predict(df_test[['Clump Thickness', 'Cell Size']])
acc = metrics.accuracy_score(df_test['Type'], predict)
precision = metrics.precision_score(df_test['Type'], predict)
accuracy = lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])
print("accuracy is: ", accuracy)
print("acc is: ", acc)
print("precision is: ", precision)
print("predict is: ", predict)

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly, c='blue')

plt.show()

# import pandas as pd
# import numpy as np
#
# columns_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
#                  'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
#                  'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# data = pd.read_csv(
#     'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
#     names=columns_names)
# data = data.replace(to_replace='?', value=np.nan)
# data = data.dropna(how='any')
# print(data.shape)






