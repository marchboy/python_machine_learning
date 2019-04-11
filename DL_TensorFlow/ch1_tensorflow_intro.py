# -*- coding: utf-8 -*-
 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义生成loss可视化的函数
plotdata = {"batchsize":[], "loss":[]}
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    # a_list = []
    # for idx, val in enumerate(a):
    #     if idx < w:
    #         a_list.append(val)
    #     else:
    #         val = sum(a[(idx-w): idx]) / w
    #         a_list.append(val)
    # return a_list
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# 生成模拟数据集
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

# 图行显示
# plt.plot(train_X, train_Y, 'ro', label='Original Data')
# plt.legend()
# plt.show()

# 重置图
tf.reset_default_graph()

# -----------------创建模型-----------------
# 占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')

# 模型参数
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# 前向结构
z = tf.multiply(W, X) + b

# 反向优化,梯度下降
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
# ---tf.global_variables_initialize
# ---必须在所有变量和模型节点op定义完成之后,才能保证定义的内容有效

init = tf.global_variables_initializer()

# 定义学习参数
training_epochs = 20
display_step = 2

# 保存模型, max_to_keep=1表示在迭代过程中之保存一个文件,新模型会覆盖旧的
saver = tf.train.Saver(max_to_keep=2)
saver_dir = './DL_TensorFlow/logs/'

# 启动图
with tf.Session() as sess:
    sess.run(init)

    # 向模型中输入参数
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            # 注入机制,将具体的实参注入到placeholder中,feed只在调用它的方法内有效,方法结束feed消失
            # 使用feed机制将具体数值通过占位符传入
            sess.run(optimizer, feed_dict={X:x, Y:y})
        
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        print("Epoch: ", epoch+1, "cost: ",loss, "W:", sess.run(W), "b:", sess.run(b))

        if not(loss == 'Na'):
            plotdata['batchsize'].append(epoch)
            plotdata['loss'].append(loss)
        saver.save(sess, saver_dir+'linearmodel.cpkt', global_step=epoch)

    print('Finish!')
    print('cost=',sess.run(cost, feed_dict={X:train_X, Y:train_Y}), "W=", sess.run(W), 'b=', sess.run(b))

    # 显示模型
    # plt.figure()
    plt.subplot(1,2,1)
    plt.plot(train_X, train_Y, 'ro', label='Original Data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    # plt.show()
    
    print(plotdata)
    plotdata['avgloss'] = moving_average(plotdata['loss'])
    # plt.figure(1)
    plt.subplot(1,2,2)
    plt.plot(plotdata['batchsize'], plotdata['loss'], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run VS. Training loss')
    plt.show()

# ----------------------------------------------
print('-------重启一个Session, 载入检查点----')
load_epoch = 19
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, saver_dir+'linearmodel.cpkt-'+ str(load_epoch))
    print('X=0.2, z=', sess.run(z, feed_dict={X:0.2}))


