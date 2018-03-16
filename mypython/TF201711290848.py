# -*- coding: utf-8 -*-
'''
Created on 2017年11月29日
@author: Administrator
'''
import tensorflow as tf
import numpy as np
# 创建一个常量op
# random1=np.random.random((2,3))
# print(random1.shape)
# m1=tf.constant([[1,3,5]])
# print(m1,type(m1))
# 创建一个常量op
# m2=tf.constant([[2],[3],[3]])
# print(m2)
# 创建一个矩阵相乘，m1,m2
# product=tf.matmul(m1,m2)
# print(product)
# 定义一个绘画，启动默认图
# sess=tf.Session()
# 调用sess的run方法执行矩阵乘法，run触发了上面3个op，到这里才开始执行运算
# result=sess.run(product)
# print(result)
# sess.close()
# with tf.Session() as sess:
#     result=sess.run(product)
#     print(result)
x=tf.Variable([1,2 ])
a=tf.constant([3,3])
# 增加一个减法op
# sub=tf.subtract(x,a)
# 增加一个加法op
# add=tf.add(x,sub)
y=tf.add(x,a)

# y=x赋值
# updata=tf.assign(x,y)
# 
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(5):
#         sess.run(updata)
#         print(sess.run(y))
# 
# 
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(5):
#         x=tf.add(x,a)
#         z=sess.run(x)
#         print(z)
# fetch
# a=tf.constant(2)
# b=tf.constant(1)
# c=tf.constant(3)
# sub=tf.subtract(a,b)
# mat=tf.multiply(a,c)
# with tf.Session() as sess:
#     result=sess.run([sub,mat])
#     print(result)
                     
# feed
# 创建占位符
# input1=tf.placeholder(tf.float32)
# input2=tf.placeholder(tf.float32)
# mul=tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run(mul,feed_dict={input1:[1],input2:[2]}))
# 字典

x_data=np.random.random(100)
y_data=x_data*345+123
b=tf.Variable(float(np.random.random(1)))
k=tf.Variable(float(np.random.random(1)))
y=k*x_data+b

# 二次代价函数
loss=tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降法来进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.1)
# 最小化代价函数
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(3000):
        sess.run(train)
        if step % 100 == 0:
            print(step,sess.run([k,b]),sess.run(loss))












    
    