#-*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

res = model.evaluate(x_test, y_test)
print(res)
print(x_test, y_test)
"""


# 导入mnist数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 探索性数据分析
print(set(train_labels))
print(type(train_images))
print(train_images.shape)
print(train_images[100].shape)



# 预处理数据
plt.figure()
plt.imshow(train_images[51500])
plt.colorbar()
plt.grid()
plt.show()

# 归一化
train_images =  train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    # 使用plt.yticks设置y轴刻度以及名称：刻度为[-2, -1.8, -1, 1.22, 3]；
    # 对应刻度的名称为[‘really bad’,’bad’,’normal’,’good’, ‘really good’]
    # https://morvanzhou.github.io/tutorials/data-manipulation/plt/2-3-axis1/
    plt.yticks([])
    plt.grid()
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# 构建模型
model = keras.Sequential(
    [
        #像素扁平化：将图像格式从二维数组（28*28像素）转化为一维数组（28*28=784像素），改层只改变数据格式
        keras.layers.Flatten(input_shape=(28,28)),
        #密集连接或者全连接神经层，该层表示具有128个节点（神经元）
        keras.layers.Dense(128, activation=tf.nn.relu), 
        # 具有10个节点的softmax层，返回一个具有10 个概率的得分数组,其总和为1，每个节点表示当前图像的属于10个类别中某一个概率。
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)

# 编译模型
model.compile(
    # 优化器，根据模型看到的数据及其损失函数更新模型的方式
    optimizer=tf.train.AdamOptimizer(),
    # 损失函数，衡量在训练期间模型的准确率
    loss = 'sparse_categorical_crossentropy',
    #指标，监控训练测试步骤，该示例使用准确率，即图像被正确分类的比例。
    metrics = ['accuracy']
)


# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 模型评估
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


# 做出预估
predictions = model.predict(test_images)
print(predictions[0])

print(np.argmax(predictions[0]))
print(class_names[np.argmax(predictions[0])])
test_labels[0]

# 绘图
def plot_image(i, predictions_array, true_label, img):
    """
    绘制图中左侧的样本图
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap = plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label], 100*np.max(predictions_array),
                            class_names[true_label]),  color=color)


def plot_value_array(i, predictions_array, true_label):
    """
    绘制图中右侧的柱状图
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


for i in range(3):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i+10, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions, test_labels)
    plt.show()


# 
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(4*num_cols, 2*num_rows)) # 画布大小

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

plt.show()

# Grab an image from test 
# 从测试集中抽取一张照片出来

img = test_images[0]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

# Predict zhis picture
# 预测这张图
res = model.predict(img)
print(res)

print(test_images[0])
plot_value_array(0, res, test_labels)
# x轴标签
plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(res[0])