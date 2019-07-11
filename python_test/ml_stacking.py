# -*- coding:utf-8 -*-

# date:2019-04-30
# Model Stacking
# https://www.cnblogs.com/jiaxin359/p/8559029.html
# http://wulc.me/2018/01/21/stacking%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3%E5%8F%8A%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0/


import numpy as np
from sklearn.model_selection import KFold


def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    构建Stacking，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test数据类型为numpy.ndarray
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set, second_level_test_set = np.zeros((train_num, )), np.zeros((test_num, ))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    print("-------------------------------------------")
    print(second_level_train_set.shape, second_level_test_set.shape, test_nfolds_sets.shape)

    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        print(i, train_index, test_index)
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, _ = x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        print("==============================================")
        print(second_level_train_set)
        test_nfolds_sets[:, i] = clf.predict(x_test)
        print(test_nfolds_sets)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


# 使用5个分类算法
"""
BaseModel之间的性能差异不能太大，表现差的模型会拖后腿；
BaseModel之间的相关性尽可能地小，从而能弥补模型的优势
"""

from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier
)
from sklearn.svm import SVC

rf_model = RandomForestClassifier()
ada_model = AdaBoostClassifier()
gbdc_model = GradientBoostingClassifier()
ext_model = ExtraTreesClassifier()
svm_model = SVC()


# 加载数据集

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris =  load_iris()
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

train_sets, test_sets = [], []
for clf in (rf_model, ada_model, gbdc_model, ext_model, svm_model):
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis =1)


# 次级分类器
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(meta_train, train_y)


dt_predict = dt_model.predict(meta_test)

print("-----------------Ensumble Result-------------------------")
print(dt_predict)
