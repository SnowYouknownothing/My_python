# -*- coding: utf-8 -*-
'''
Created on 2017年11月29日
@author: Administrator
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 使用numpy生成200个随机数
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
# print(x_data)
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

# y_true=np.square(x_data)
# plt.title("tf")
# plt.xlim(xmax=0.5,xmin=-0.5)
# plt.ylim(ymax=0.3,ymin=0)
# plt.xlabel("x_data")
# plt.ylabel("y_data")
# plt.scatter(x_data,y_data,s=5,alpha=0.4,marker='o')
# plt.scatter(x_data,y_true,s=5,alpha=0.8,marker='o')
# plt.show()  

# 定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_L1=tf.matmul(x,weights_L1)+biases_L1
output_L1=tf.nn.tanh(Wx_plus_L1)

weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_L2=tf.matmul(output_L1,weights_L2)+biases_L2
perdiction=tf.nn.tanh(Wx_plus_L2)

loss=tf.reduce_mean(tf.square(y-perdiction))
# 二次代价函数

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 最小化代价函数

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#     变量初始化
    for step in range(3000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
#         if step%600==0:
#             print(step,sess.run(loss),sess.run(perdiction))
    perdiction_value=sess.run(perdiction,feed_dict={x:x_data})
    
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,perdiction_value,'r-',lw=3)
    plt.show()
    print(sess.run(weights_L1))



