# -*- coding: utf-8 -*-

# function test 
# date:2019-04-16

# ------------------------------------基础概念----------------------------------------------

"""
enbedding在数学上表示一个mapping，即f:x->y，对应为一个function，该funciton是：
1、injective （单射函数，每个Y只有唯一的X对应，反之亦然）；
2、structure-preserving（结构保存， 即在x所属空间上的x1 < x2， 映射后在y的空间上映是y1 < y2）

那么对于word embedding，就是将单词word映射到另外一个空间，这个映射具有injective和structure-preserving的特点。
即词嵌入，把X所属空间的单词映射为到Y空间的多维向量，那么该多维向量相当于嵌入到Y所属空间中，一个萝卜一个坑。

word embedding，就是找到一个映射或者函数，生成在一个新的空间上的表达，该表达就是word representation。
"""

# batch、epoch、iteration
# 梯度（gradient）：斜率、斜坡的倾斜度
# 下降（Descent）：代价/损失函数的下降

# https://www.jiqizhixin.com/articles/2017-09-25-3
# epoch：当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个 epoch；
# batch size：一个 batch 中的样本总数。batch size 和 number of batches 是不同的；
# 在不能将数据一次性通过神经网络的时候，就需要将数据集分成几个 batch。
# 对于一个有 2000 个训练样本的数据集。将 2000 个样本分成大小为 500 的 batch，
# 那么完成一个 epoch 需要 4 个 iteration.

# The regularization term is what people usually forget to add. 
# The regularization term controls the complexity of the model, 
# which helps us to avoid overfitting.
# The tree ensemble model consists of a set of classification and regression trees (CART).
# an important fact is that the two trees try to complement each other.


# How should we learn the trees? 
# The answer is, as is always for all supervised learning models: 
# define an objective function and optimize it!

# It is intractable to learn all the trees at once.
#  Instead, we use an additive strategy: fix what we have learned, and add one new tree at a time.

# ------------------------------------训练方法----------------------------------------------
"""
对于抽取的B个样本，其中B=PK，P表示P个身份的人，K表示每个身份的人有K张图片


"""


