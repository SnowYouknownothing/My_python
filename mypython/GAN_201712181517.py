# -*- coding: utf-8 -*-
'''
Created on 2017年12月18日
@author: Administrator
'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_digits
digits=load_digits()
x_test=digits.data
print(x_test.shape)
y_test=digits.target

index_find=[]
for i in range(len(y_test)):
    if y_test[i]==4:
        index_find.append(x_test[i])
index_find=np.array(index_find)
print(index_find.shape)
        
x_test=index_find
# x_test=np.reshape(x_test,[181,8,8])
# 
# plt.imshow(x_test[0]*20)
# plt.show()



# minst=input_data.read_data_sets('minst_data')
# x_test=minst.test.images/16
# y_test=minst.test.labels*0.1
# print(x_test.shape,y_test.shape)
# print(y_test[4])
# 
# plt.imshow(x_test[4])
# plt.show()
 
with tf.variable_scope('G'):
    x_1=tf.placeholder(tf.float32,[None,64],name='x_real_input')
    G_mid1=tf.layers.dense(x_1,5000,tf.nn.relu,name='G_mid1')
    G_mid2=tf.layers.dense(G_mid1,5000,tf.nn.relu,name='G_mid2')
    G_out=tf.layers.dense(G_mid2,64)
  
with tf.variable_scope('D'):
    x_2=tf.placeholder(tf.float32,[None,64],name='x_input')
    D_real_mid1=tf.layers.dense(x_2,5000,tf.nn.relu,name='1')
    D_real_mid2=tf.layers.dense(D_real_mid1,5000,tf.nn.relu,name='2')
    D_real_out=tf.layers.dense(D_real_mid2,1,tf.nn.sigmoid,name='3')
      
    D_fake_mid1=tf.layers.dense(G_out,5000,tf.nn.relu,name='1',reuse=True)
    D_fake_mid2=tf.layers.dense(D_fake_mid1,5000,tf.nn.relu,name='2',reuse=True)
    D_fake_out=tf.layers.dense(D_fake_mid2,1,tf.nn.sigmoid,name='3',reuse=True)
      
# loss_D=-tf.reduce_mean(tf.log(D_real_out)+tf.log(1-D_fake_out))
# loss_G=tf.reduce_mean(tf.log(1-D_fake_out))

loss_G=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_fake_out)*(1-0.1),
                                                           logits=D_fake_out))

d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_real_out)*(1-0.1),
                                                              logits=D_real_out))

d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_fake_out),
                                                              logits=D_fake_out))

loss_D=tf.add(d_loss_real,d_loss_fake)  

  
train_step_D=tf.train.AdamOptimizer(1e-6).minimize(loss_D,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D')) 
train_step_G=tf.train.AdamOptimizer(1e-6).minimize(loss_G,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='G'))

a,b=[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_fake=np.random.randn(181,64)
    for epoch in range(5000):       
        D_real_out1,D_fake_out1,loss_D1,loss_G1,G_out1=sess.run([D_real_out,D_fake_out,loss_D,loss_G,G_out,train_step_D,train_step_G],
                                                              {x_1:x_test[:100],x_2:x_fake[:100]})[:5]
  
# x_test_re=np.reshape(x_test*16,[10000,28,28])
        a.append(loss_D1)
        b.append(loss_G1)
G_out1_re=np.reshape(G_out1,[100,8,8])
  
# plt.figure()
# plt.plot(loss_D1)
# plt.plot(loss_G1)
# plt.show()
  
print(D_real_out1[-1],D_fake_out1[-1],loss_D1,loss_G1)
 
# plt.imshow(x_test_re[4])
plt.figure()
plt.plot(a)
plt.plot(b)
plt.show()


for i in range(10):
    plt.figure()
    plt.imshow(G_out1_re[i*10])
    plt.show()    
            
print(G_out1_re[2])
print(x_test[0])       
