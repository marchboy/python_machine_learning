# -*- coding：utf-8 -*-

import pandas as pd
import numpy as np

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import classification_report

columns_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=columns_names)

pd.set_option('display.width',1000)  #设置显示的最带宽度

# print(data.describe())
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
# print(data.shape)
# print(data.describe())

X_train, X_test, y_train, y_test = train_test_split(data[columns_names[1:10]], data[columns_names[10]], test_size=0.25, random_state=33)
# print(y_train.value_counts())
# print(y_test.value_counts())
# print(X_train[1:5], '\n',X_test[1:5])

# 标准化
"""
二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）；
fit_transform(partData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该partData进行转换transform，从而实现数据的标准化、归一化等等；
根据对之前部分fit的整体指标，对剩余的数据（restData）使用同样的均值、方差、最大最小值等指标进行转换transform(restData)，从而保证part、rest处理方式相同；
必须先用fit_transform(partData)，之后再transform(restData)；
如果直接transform(partData)，程序会报错；
如果fit_transfrom(partData)后，使用fit_transform(restData)而不用transform(restData)，虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。

https://blog.csdn.net/anecdotegyb/article/details/74857055
https://blog.csdn.net/quiet_girl/article/details/72517053
"""

ss = StandardScaler()
X_train = ss.fit_transform(X_train)    # 先拟合数据，然后转化它将其转化为标准形式
X_test = ss.transform(X_test)          # Perform standardization by centering and scaling（通过找中心和缩放等实现标准化）

# 初始化
lr = LogisticRegression()
sgdc = SGDClassifier()
lr.fit(X_train, y_train)  # LR分类器 训练模型
lr_y_predict = lr.predict(X_test) # 对X_test进行预测

sgdc.fit(X_train, y_train) # 随机梯度下降分类器 
sgdc_y_predict = sgdc.predict(X_test) # 对X_test进行预测

# 性能分析（Performance）
print('Accuracy of LR Classfier:', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

print('Accuracy of SGD Classfier:', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))


# sklearn分类：https://blog.csdn.net/u012526003/article/details/79054012
# sklearn中的数据预处理 http://d0evi1.com/sklearn/preprocessing/

# transform、fit_transform
from sklearn.preprocessing import MinMaxScaler

data = np.array(np.random.randint(-100, 100, 24).reshape(6,4))
print(data)
train = data[:4]
test = data[4:]

minmaxTransformer = MinMaxScaler(feature_range=(0, 1))
train_transformer = minmaxTransformer.fit_transform(train)
print(train_transformer)
test_transformer = minmaxTransformer.transform(test)
test_transformer2 = minmaxTransformer.fit_transform(test)
print(test_transformer, '\n', test_transformer2)

