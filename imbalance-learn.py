# conding = utf-8
# Comparision of the different over-sampling algrithms
#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import make_pipeline


def create_database(n_samples=1000, weights=[0.01, 0.01, 0.98], n_classes=3):
    return make_classification(
        n_samples=n_samples, n_features=2, n_informative=2,
        n_redundant=0, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1,weights=weights,
        class_sep=0.8, random_state=0
    )

# plot the sample space after resampling to illustrate the characterisitic of an algrithmn.
# 用于绘制重采样后的样本空间，以说明算法的特征。
def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_sample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    ax.spines('top').set_visible(False)
    ax.spines('right').set_visible(False)
    ax.get_xaxis().tick_buttom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)

# plot the decision function of a classfier given 
# 用于绘制给定数据的分类器的决策函数。
def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # print(x_min, x_max, y_min, y_max)
    # meshgrid: https://zhuanlan.zhihu.com/p/29663486 
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step),
        np.arange(y_min, y_max, plot_step)
    )
    print(xx.shape)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # ravel: https://blog.csdn.net/liuweiyuxiang/article/details/78220080
    # ravel 降维到一维，区别于flatten
    # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')


if __name__ == '__main__':
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    # ax_arr = (ax1, ax2, ax3, ax4)
    # print(ax_arr)
    # weights_arr = ((0.01, 0.01, 0.98), (0.01, 0.05, 0.94),
    #             (0.2, 0.1, 0.7), (0.33, 0.33, 0.33))

    # for ax, weights in zip(ax_arr, weights_arr):
    #     X, y = create_database(n_samples=1000, weights=weights)
    #     clf = LinearSVC()
    #     clf.fit(X, y)
    #     plot_decision_function(X, y, clf, ax)
    #     ax.set_title("Linear SVC with y={}".format(Counter(y)))
    # fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    X, y = create_database(n_samples=10000, weights=(0.01, 0.05, 0.94))
    clf = LinearSVC()
    clf.fit(X,y)
    plot_decision_function(X, y, clf, ax1)
    ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
    pipe = make_pipeline(RandomOverSampler(random_state=0), LinearSVC())
    pipe.fit(X, y)
    plot_decision_function(X, y, pipe, ax2)
    ax2.set_title('Decision fuction for RandomOverSampler')
    fig.tight_layout()

# ros = RandomOverSampler(random_state=0)   # 分类样本数过采样,自然随机过采样
# X_resampled, y_resampled = ros.fit_sample(X, y)
# print(y_resampled)
# print('-' * 50)

# print(sorted(Counter(y_resampled).items()))

# print(X[1:10], "\n")
# print(y)
# print(Counter(y))

# clf = LinearSVC()
# clf.fit(X_resampled, y_resampled)
