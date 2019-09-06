# -*- coding: utf-8 -*-


#%%

import numpy as np
import numpy.linalg as LA

mat = np.matrix([[1,2],[3,4]])
inv_mat = np.linalg.inv(mat)  # 矩阵的逆
# print(inv_mat)


# 研究范数理论的意义：研究数值方法的收敛性、稳定性及误差分析等问题时，范数理论显得十分重要
# 通俗来说就是距离

# 向量范数

def vector_norm():
    a = np.arange(9)-5
    print(a)
    print("向量0范数：", LA.norm(a, 0))        # 向量中非零元素的个数
    print("向量1范数：", LA.norm(a, 1))        # 向量元素绝对值之和
    print("向量2范数：", LA.norm(a, 2))        # 向量元素绝对值的平方和再开方， 可以引申到p范数

    print("向量无穷范数：", LA.norm(a, np.inf))     # 向量所有元素绝对值中的最大值
    print("向量负无穷范数：", LA.norm(a, -np.inf))  # 所有元素绝对值中最小值

def matrix_norm():
    a = np.arange(9)-5
    b = a.reshape(3,3)
    
    b_t = np.transpose(b)
    b_new = np.dot(b_t, b)
    b_new_1 = np.dot(b, b_t)
    x = np.linalg.eigvals(b_new)
 
    print(b)
    print("矩阵的转置：", b_t, '\n\n矩阵的转置乘矩阵：\n' ,b_new, '\n\n矩阵乘矩阵的转置:\n' ,b_new_1, '\n\n矩阵的特征值:\n', x)
    
    print("矩阵1范数：", LA.norm(b, 1))       # （列和范数）矩阵所有列向量绝对值之和的最大值
    print("矩阵无穷范数：", LA.norm(b, np.inf))  # （行和范数）矩阵所有行向量绝对值之和的最大值

    print("矩阵2范数(矩阵最大特征值的开方)：", LA.norm(b, 2))       # 最大特征值开方
    print("矩阵F范数：", LA.norm(b, 'fro'))                       # 矩阵所有元素绝对值的平方和再开平方，是向量范数的推广(AA^T矩阵迹的开方)。

if __name__ == "__main__":
    vector_norm()
    print("-----------------------------------------")
    matrix_norm()



# 一个向量空间所包含的最大线性无关向量的数目，称作该向量空间的维数(向量空间中不重叠的向量的数量)






