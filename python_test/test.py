#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
##--maxn.py
#寻找一组数据当中最大者
def main():
    n = eval(input("How many numbers are there?"))
    max = eval(input("Enter a number >> "))
    for i in range(n-1):
        x = eval(input("Enter a number >> "))
        if x > max:
            max = x
        print("The largest value is ", max)
main()


import pandas as pd
import os
os.getcwd()
path = 'E:\\MyStudy\\Python\\pydata_book_master\\ch02\\names'
os.chdir(path)

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    filename = 'yob%d.txt' % year
    frame = pd.read_csv(filename, names=columns)
    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces, ignore_index= True)
print(len(names))


# A, B, C = input().split(' ')
# print("A + B + C = ", int(A) + int(B) + int(C))


# http://bbs.fishc.com/thread-79821-1-1.html
# https://blog.csdn.net/ly_ysys629/article/details/55224284
# https://blog.csdn.net/xw_classmate/article/details/51333646  讲的不错

# loc[:,['tip','total_bill']]才能索引列，
# loc通过行标签索引行数据，即通过index和columns的值进行选取。loc方法有两个参数，按顺序控制行列选取。
# iloc：通过行号选取数据，即通过数据所在的自然行列数为选取数据。iloc方法也有两个参数，按顺序控制行列选取
# ix:混合索引，同时通过标签和行号选取数据。df.ix[1:3,['b','c']]
# at/iat：通过标签或行号获取某个数值的具体位置。


# 总结：
# 1）.loc,.iloc,.ix,只加第一个参数如.loc[[1,2]],.iloc([2:3]),.ix[2]…则进行的是行选择---- 
# 2）.loc,.at，选列是只能是列名，不能是position---- .iloc[:,[1]], 
# 3）.iloc,.iat，选列是只能是position，不能是列名 
# 4）df[]只能进行行选择，或列选择，不能同时进行列选择，列选择只能是列名。
"""

import pandas as pd
import numpy as np
import random

df = pd.DataFrame(
    {"total_bill":[16.99, 10.34, 23.68, 23.68, 24.59],
    "tip":[1.01, 1.66, 3.50, 3.31, 3.61],
    "sex":['Female', 'Male', 'Male', 'Male', 'Female']})

print(df.dtypes, '\n')
print(df.index, '\n')
print(df.columns, '\n')
print(df.values, '\n')
print(df)

print(df.loc[1:3,['total_bill', 'tip']])
print(df.loc[0:3, 'tip':'total_bill'])

print(df.iloc[1:3, [1,2]])
print(df.iloc[1:3, 1:3])
print(df.loc[1:3], '\n', df.iloc[1:3])

print('-' * 50)


rnd_1 = [random.randrange(1,20) for x in range(1000)]
rnd_2 = [random.randrange(1,20) for x in range(1000)]
rnd_3 = [random.randrange(1,20) for x in range(1000)]
fecha = pd.date_range('2012-4-10', '2015-1-4')

data = pd.DataFrame({'fecha':fecha, 'rnd_1':rnd_1, 'rnd_2':rnd_2, 'rnd_3':rnd_3})
# print(data)
print(data.describe())

print(data[0:10][["fecha", "rnd_1"]])
print(data[0:10])
print('-' * 50)
print(data.loc[1:5])  # 推荐此方法进行行列选择，而不是[],注意与iloc的区别



def vector_norm():
    a = np.arange(9) - 4
    print(a)
    print(np.linalg.norm(a, 0)) # L0范数，向量中非0元素的个数
    print(np.linalg.norm(a, 1)) # 向量所有元素的绝对值之和
    print(np.linalg.norm(a, 2)) # 欧式距离，P范数
    print(np.linalg.norm(a, -np.inf)) # 所有向量元素中的最小值
    print(np.linalg.norm(a, np.inf))  # 所有向量元素中的最大值

def matrix_norm():
    a = (np.arange(9) - 4).reshape(3,3) 
    a = np.matrix(a)
    print(a)
    a_t = a.T
    _dot = np.dot(a_t, a)
    characteristic_value = np.linalg.eigvals(_dot)
    print(characteristic_value)
    # print(np.linalg.norm(a, 0))  # 
    print(np.linalg.norm(a, 1))  # 列和范数，矩阵列向量中绝对值之和的最大值
    print(np.linalg.norm(a, 2))  # 谱范数，计算方法为A'A矩阵的最大特征值的开平方
    print(np.linalg.norm(a, np.inf)) # 行和范数, 矩阵行向量中绝对值之和的最大值
    print(np.linalg.norm(a, "fro"))  # Frobenius范数，矩阵元素的绝对值的平方和再开方

if __name__ == '__main__':
    print('-'*50)
    print('vector_norm: \n')
    print(vector_norm())
    print('matrix_norm: \n')
    print(matrix_norm())


from collections import namedtuple
t = namedtuple('d',['x','y'])
tt = t(1,2)
print(tt.x, tt.y)

from types import SimpleNamespace
p = SimpleNamespace(x=1, y=3)
print(p.x, p.y)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_test = [1,2,3,3,3,3]
y_pred = [1,3,3,2,3,2]
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




# 散沙---二维数组与矩阵操作
import numpy as np

#-------------二维数组----------------
arr2d_random  = np.random.rand(10).reshape(5,2)
arr2d_zeros = np.zeros_like(arr2d_random)
arr2d_ones = np.ones_like(arr2d_random)

print(arr2d_random, '\n\n', arr2d_zeros, '\n\n' ,arr2d_ones)
print(np.zeros((5,2)), '\n'*2, np.ones([5,2]), '\n'*2, np.eye(3))

# 使用函数传入下标作为参数，生成二维数组
def CameronThePrimeMinister(i, j):
    return (i+1)*(j+1)

print(np.fromfunction(CameronThePrimeMinister, (8,8)))

# 使用广播broadcasting
x = np.arange(1, 10)
print(x.reshape(-1, 1), '\n'*2, x, '\n'*2, x.reshape(-1,1) * x)


(x, y) = np.ogrid[0:1:6j, 0:1:6j]  # 第三个参数带j即虚数，表示要返回的数组的长度；实数表示步长；mgrid则返回广播后的数组
print(np.exp(-x**2 - y**2))

# 使用笛卡儿积，outer关键字
np.multiply.outer(range(1, 10), range(1, 10))

#-------------矩阵----------------
print(arr2d_random, '\n'*2, arr2d_random.T)
cov = np.cov(arr2d_random.T)
print(cov)

stdiag = np.diag(np.sqrt(np.diag(cov)))
print(stdiag)

invstdiag = np.array(np.mat(stdiag).I)
print(invstdiag)

corr = np.mat(stdiag).I * np.mat(cov) * np.mat(stdiag).I
print(corr)

print(np.corrcoef(arr2d_random.T))

#----------------利用矩阵实现最小二乘法解线性回归（OLS）-------------------
#
x = np.random.rand(30).reshape(15, 2)
y = x[:,0] * 0.7 - x[:,1] * 0.2 + 0.1 + 0.1*np.random.rand(15)

Xcvr = np.mat(np.hstack((np.ones(15).reshape(15,1), x)))
print(Xcvr)


H = Xcvr * (Xcvr.T * Xcvr).I * Xcvr.T
betahats = np.dot((Xcvr.T * Xcvr).T * Xcvr.T, y)

preds = np.dot(H, y)

print(betahats, '\n'*2, preds)

import statsmodels.formula.api as sm
model = sm.OLS(y, Xcvr).fit()
print(model.summary())


# PCA 数据降维

import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, scale

# x = np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
# y = np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])


x_mean = np.mean(x)
y_mean = np.mean(y)

scaled_x = x - x_mean
scaled_y = y - y_mean
# data = np.matrix([[scaled_x[i], scaled_y[i]] for i in range(len(scaled_x))])
data = np.matrix(list(zip(scaled_x, scaled_y)))

standard = StandardScaler()
standard = scale()
data_standard = standard.fit_transform(np.array(list(zip(x,y))))

plt.scatter(scaled_x, scaled_y)
plt.scatter(x, y)
plt.show()

cov = np.cov(scaled_x, scaled_y)
cov = np.cov(data.T)

eig_val, eig_vec = np.linalg.eig(cov)
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)

feature=eig_pairs[0][1]
new_data_reduced=np.transpose(np.dot(feature,np.transpose(data)))
print(new_data_reduced)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
data = np.matrix(list(zip(x,y)))
new_data = pca.fit_transform(data)
print(new_data)
print(pca.explained_variance_ratio_, '\n'*2, pca.explained_variance_)




import uuid
name = 'test_name'
namespace = 'test_namespace'
print(uuid.uuid1())
print(uuid.uuid3(uuid.NAMESPACE_DNS, "myString"))
print(uuid.uuid4())
print(uuid.uuid5(uuid.NAMESPACE_DNS, "myString"))

import heapq
portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]

cheap = heapq.nsmallest(3, portfolio, key=lambda s:s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s:s['price'])
print(cheap, expensive)


nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
heap = list(nums)
heapq.heapify(heap)
print(heapq.heappop(heap))
print(heapq.heappop(heap))



from collections import defaultdict

def result():
    return defaultdict(set)

d = defaultdict(result)
d['a']['b'].add('100')
d['a']['b'].add('1400')
d['a']['c'].add('100')

