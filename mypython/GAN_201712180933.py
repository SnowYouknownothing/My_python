# -*- coding: utf-8 -*-
'''
Created on 2017年12月18日
@author: Administrator
'''
# from sklearn.datasets import load_digits
# from numpy import reshape
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# x=load_digits()
# x1=x.data
# x2=x.images
# x3=x.target
# print(x1.shape,x2.shape,x3.shape)
# print(x2[3],x1[3])
# 
# x4=reshape(x1[3],[8,8])
# print(x4)
# 
# import pylab as pl
# pl.gray()
# pl.matshow(x2[3])
# pl.matshow(x4)
# pl.show()

x_real=np.vstack(np.linspace(-1, 1, 15) for _ in range(1))
y_real=x_real**2
# y_rea2=y_real[:np.newaxis]
# plt.figure
# plt.plot(x_real,y_real)
# plt.show()

# x_fake=np.linspace(0,2,20)[:np.newaxis]
print(x_real.shape,y_real.shape)


# x_2=tf.placeholder(tf.float32,[None,1])

with tf.variable_scope('G'):
    x_1=tf.placeholder(tf.float32,[1,15])
    dense1=tf.layers.dense(x_1,1280, tf.nn.relu)
    dense2=tf.layers.dense(dense1,1280, tf.nn.relu)
    G_out=tf.layers.dense(dense2,15)
with tf.variable_scope('D'):
    y=tf.placeholder(tf.float32,[1,15],name='real_in')
    dense3=tf.layers.dense(y,1280,tf.nn.relu,name='1')
    dense4=tf.layers.dense(dense3,1280,tf.nn.relu,name='2')
    G_real=tf.layers.dense(dense4,1,tf.nn.sigmoid,name='3')
    
    dense31=tf.layers.dense(G_out,1280,tf.nn.relu,name='1',reuse=True)
    dense41=tf.layers.dense(dense31,1280,tf.nn.relu,name='2',reuse=True)
    G_fake=tf.layers.dense(dense41,1,tf.nn.sigmoid,name='3',reuse=True)

# D_loss = -tf.reduce_mean(tf.log(G_real) + tf.log(1-G_fake))
# G_loss = tf.reduce_mean(tf.log(1-G_fake))

G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(G_fake),
                                                           logits=G_fake))

d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(G_real),
                                                              logits=G_real))

d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(G_fake),
                                                              logits=G_fake))

D_loss=tf.add(d_loss_real,d_loss_fake) 


train_step1=tf.train.AdamOptimizer(0.00001).minimize(D_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D'))
train_step2=tf.train.AdamOptimizer(0.00001).minimize(G_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='G'))
i_index,j_index=[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_fake=np.random.randn(1,15)
    for epoch in range(1000):
        G_out1,G_real1,G_fake1,D_loss1,G_loss1=sess.run([G_out,G_real,G_fake,D_loss,G_loss,train_step1,train_step2],{x_1:x_fake,y:y_real})[:5]
#         if epoch % 600==0:
        i_index.append(D_loss1)
        j_index.append(G_loss1)
#         if epoch%100==0:
#             plt.figure()
#             plt.plot(x_real[0],y_real[0])
#             plt.plot(x_real[0],G_out1[4])
#             plt.show() 
#             plt.figure()
#             plt.plot(i_index)
#             plt.plot(j_index)
    #     plt.plot(G_real1[0])
    #     plt.plot(G_fake1[0])
#             plt.show()
plt.figure()
plt.plot(x_real[0],y_real[0])
plt.plot(x_real[0],G_out1[0])
plt.show() 
print(D_loss1,G_loss1)
print(G_real1[0],G_fake1[0])












