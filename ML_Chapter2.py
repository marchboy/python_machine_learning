# -*- coding：utf-8 -*-

import pandas as pd
import numpy as np

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
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
print(X_train[1:5], X_test[1:5])

# 标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
print(X_train, X_test)

# 初始化
lr = LogisticRegression()
sgdc = SGDClassifier()
lr.fit(X_train, y_train)  # LR分类器 训练模型
lr_y_predict = lr.predict(X_test) # 对X_test进行预测

sgdc.fit(X_train, y_train)  # 随机梯度下降分类器 
sgdc_y_predict = sgdc.predict(X_test) # 对X_test进行预测

# 性能分析（Performance）
print('Accuracy of LR Classfier:', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

print('Accuracy of SGD Classfier:', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
