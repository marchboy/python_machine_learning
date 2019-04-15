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

