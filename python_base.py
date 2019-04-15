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


<<<<<<< HEAD
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

=======
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
>>>>>>> 44f4f9e1ba7f82ee60b5f7405e7abea2966e3646

