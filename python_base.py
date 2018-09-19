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


