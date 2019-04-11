# -*- coding: utf-8 -*-

"""
能自动利用CPU的多线程进行并行运算
"""

import pandas as pd

titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
print(titanic)
titanic.columns.values
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

X['age'].fillna(X['age'].mean(), inplace=True)

