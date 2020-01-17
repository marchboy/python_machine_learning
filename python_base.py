# -*- coding: utf-8 -*-

import numpy as np

# def isscalar(num):
#     if isinstance(num, np.generic):
#         return True
#     else:
#         return False
print(np.isscalar(3.2))
print(np.isscalar([3.3]))
print(np.isscalar(False))


x = np.matrix([[1,2],[5,6]])
print(x)
a, b = x.mean(0), x.mean(1)
print(a, b)
c = a -b
print(c)


import torch
a = torch.Tensor([10])
print(a, type(a))

t = torch.Tensor(5,3)
print(t)
print(t.shape)

q = torch.Tensor(4,4)
p = torch.Tensor(4,4)
ones = torch.ones(4,4)
print(p, q, ones)

print(p+q)
print(p-ones)
print(p * ones)
print(q / ones)


import pandas as pd

students = [ ('jack', 34, 'Sydeny' , 'Australia') ,
             ('Riti', 30, 'Delhi' , 'India' ) ,
             ('Vikas', 31, 'Mumbai' , 'India' ) ,
             ('Neelu', 32, 'Bangalore' , 'India' ) ,
             ('John', 16, 'New York' , 'US') ,
             ('Mike', 17, 'las vegas' , 'US')  ]

dfObj = pd.DataFrame(students, columns=['Name','Age','City','Country'], index=['a', 'b', 'c' , 'd' , 'e' , 'f'])
print(dfObj)
columnsNameArr = dfObj.columns.values  # transfer to ndarray 
listOfColumnNames = list(columnsNameArr)

indexNameArr = dfObj.index.values
listOfindexNameArr = list(indexNameArr)

#-------------------------------------------------------------

import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array(np.random.randint(-100,100,24).reshape(6,4))
train = data[:4]
test = data[4:]

minmaxTransformer = MinMaxScaler(feature_range=(0,1))

train_transformer = minmaxTransformer.fit_transform(train)
test_transformer = minmaxTransformer.transform(test)

print(train_transformer)
print(test_transformer)

import tensorflow as tf
input = tf.Variable(tf.constant(1.0, shape=[1,5,5,1]))

op1_max_pooling_same = tf.nn.max_pool(input, [1,2,2,1], strides=[1,2,2,1], padding='SAME')
op2_max_pooling_valid = tf.nn.max_pool(input, [1,2,2,1], strides=[1,2,2,1], padding='VALID')

op3_avg_pooling_same = tf.nn.avg_pool(input, [1,2,2,1], strides=[1,2,2,1], padding='SAME')
op4_global_pooling_same = tf.nn.avg_pool(input, [1,5,5,1], strides=[1,5,5,1],padding='SAME')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(init))
    print("op1_max_pooling_same:\n", sess.run(op1_max_pooling_same))
    print("op2_max_pooling_valid:\n", sess.run(op2_max_pooling_valid))
    print("op3_max_pooling_same:\n", sess.run(op3_avg_pooling_same))
    print("op4_global_pooling_same:\n", sess.run(op4_global_pooling_same))


from XXPBBase_pb2 import UserInfo
user_info = UserInfo()

user_info.ParseFromString()
print(user_info)