# -*- coding: utf-8 -*-
'''
Created on 2017年11月28日
@author: Administrator
'''
import tensorflow as tf
import numpy as np
x_data=np.random.rand(100).astype(np.float32)
y_data=50*x_data+30
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))
y=W*x_data+b
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for s in range(200):
    sess.run(train)
    if  s % 20 == 0:
        print(s,sess.run(W),sess.run(b))

