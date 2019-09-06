# -*- coding: utf-8 -*-




import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

print('Setting Eager Mode...')
tfe.enable_eager_execution()

a = tf.constant(2)
b = tf.constant(5)
c = a + b
print(c)


a = tf.constant([[2,3],[2,0]], dtype=tf.float32)
print('Tensor:\n a = ', a)
b = np.array([[3,0], [5,1]], dtype=np.float32)
print('NumpyArray:\n b = ', b)
print("Running operations, without tf.Session")


c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a * b = %s" % d)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])






# # 常量
# a = tf.constant(2)
# b = tf.constant(6)

# with tf.Session() as sess:
#     print("a = 2, b = 3")
#     print("addition with constants: ", sess.run(a+b))
#     print("Multiplication with constant: ", sess.run(a*b))

# # 占位符
# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)

# add = tf.add(a,b)
# mul = tf.multiply(a, b)

# with tf.Session() as sess:
#     print("Additions with variables:", sess.run(add, feed_dict={a:2, b:3}))
#     print("Multiplication with variables:", sess.run(mul, feed_dict={a:2, b:3}))
    

# # 矩阵乘法
# matrix1 = tf.constant([[3., 3.]])
# matrix2 = tf.constant([[2.],[2.]])
# product = tf.matmul(matrix1, matrix2)

# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)