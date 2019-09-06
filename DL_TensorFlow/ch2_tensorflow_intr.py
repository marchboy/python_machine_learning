# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf


hello = tf.constant("Hello Tensorflow")
sess = tf.Session()
cont =  sess.run(hello)
print(cont)
sess.close()


a = tf.constant(3)
b = tf.constant(4)
with tf.Session() as sess:
    res = sess.run(a+b)
print(res)


# 演示注入机制
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("add: {}".format(sess.run(add, feed_dict={a:3, b:4})))
    print("mul: {}".format(sess.run(mul, feed_dict={a:5, b:9})))
    # 使用注入机制获取节点
    print(sess.run([mul, add], feed_dict={a:5, b:3}))



