# -*- coding: utf-8 -*-
'''
Created on 2018年1月15日
@author: Administrator
'''
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

x=np.linspace(-10,10,20)
x_noise=np.random.randn((20))
print(x_noise)
y=3*x+10+x_noise
# plt.plot(x,y)
x_aver=np.average(x)
y_aver=np.average(y)
sum_1,sum_2=0,0
for i in range(20):
    sum_1+=(x[i]-x_aver)*(y[i]-y_aver)
    sum_2+=(x[i]-x_aver)**2
b_1=sum_1/sum_2
b_2=y_aver-b_1*x_aver
y_predition=b_1*x+b_2
 
print(b_1,b_2)
# plt.plot(x,y_predition,'r-')
# plt.scatter(x,y)
# plt.show()

# GAN

x_real=np.vstack(np.linspace(-10, 10, 20) for _ in range(1))
y_real=3*x_real+10+x_noise
print(x.shape,y.shape,x_real.shape,y_real.shape)

with tf.variable_scope('G'):
    x_1=tf.placeholder(tf.float32,[1,20])
    dense1=tf.layers.dense(x_1,128*8, tf.nn.relu)
    dense2=tf.layers.dense(dense1,128*8, tf.nn.relu)
    G_out=tf.layers.dense(dense2,20)
with tf.variable_scope('D'):
    y_1=tf.placeholder(tf.float32,[1,20],name='real_in')
    dense3=tf.layers.dense(y_1,128*8,tf.nn.relu,name='1')
    dense4=tf.layers.dense(dense3,128*8,tf.nn.relu,name='2')
    G_real=tf.layers.dense(dense4,1,tf.nn.sigmoid,name='3')
    
    dense31=tf.layers.dense(G_out,128*8,tf.nn.relu,name='1',reuse=True)
    dense41=tf.layers.dense(dense31,128*8,tf.nn.relu,name='2',reuse=True)
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
    x_fake=np.random.randn(1,20)
    for epoch in range(5000):
        G_out1,G_real1,G_fake1,D_loss1,G_loss1=sess.run([G_out,G_real,G_fake,D_loss,G_loss,train_step1,train_step2],{x_1:x_fake,y_1:y_real})[:5]
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
print(x.shape,y.shape,y_predition.shape,x_real[0].shape,G_out1[0].shape)
plt.figure()
plt.plot(x,y_predition,'r-')
plt.scatter(x,y)
plt.plot(x_real[0],G_out1[0],'g-')
plt.show() 



