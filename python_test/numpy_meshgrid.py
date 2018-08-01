#! /usr/bin/env python3
# -*- coding: utf-8 -*-


# desciption:
# [X,Y] = meshgrid(x,y)
# 将向量x和y定义的区域转换成矩阵X和Y,
# 其中矩阵X的行向量是向量x的简单复制，
# 而矩阵Y的列向量是向量y的简单复制


import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

m,n = (5,3)
x = np.linspace(0,1,m)
y = np.linspace(0,1,n)

X, Y = np.meshgrid(x, y)

print(X, '\n', Y, '\n' ,np.meshgrid(x, y))

plt.plot(X, Y, marker='.', color='blue', linestyle='none')
plt.show()

