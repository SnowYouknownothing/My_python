# -*- coding: utf-8 -*-
'''
Created on 2018年1月11日
@author: Administrator
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

minst=input_data.read_data_sets('Minst_data',one_hot=True)

x_train=minst.train.images
y_train=minst.train.labels
x_test=minst.test.images
y_test=minst.test.labels

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

filters_=32
batch_size=128
n_batch=55000//batch_size

x=tf.placeholder(tf.float32, [None,784], name='input_data')
inputs_data=tf.reshape(x, [-1,28,28,1])
y=tf.placeholder(tf.float32,[None,10],name='lables')
keep_prob_=tf.placeholder(tf.float32)
lr=tf.Variable(0.001,dtype=tf.float32)

layer1=tf.layers.conv2d(inputs_data, filters_, 3, 1, 'same',kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),name='layer1')
layer1=tf.layers.batch_normalization(layer1)
layer1=tf.maximum(0.01*layer1,layer1)
layer1=tf.nn.dropout(layer1, keep_prob_)
layer1=tf.layers.max_pooling2d(layer1, 2, 2,'same')

layer2=tf.layers.conv2d(layer1, filters_*2, 3, 1, 'same',kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),name='layer2')
layer2=tf.layers.batch_normalization(layer2)
layer2=tf.maximum(0.01*layer2,layer2)
layer2=tf.nn.dropout(layer2, keep_prob_)
layer2=tf.layers.max_pooling2d(layer2, 2, 2,'same')

layer3=tf.layers.conv2d(layer2, filters_*4, 4, 1, 'valid',kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),name='layer3')
layer3=tf.layers.batch_normalization(layer3)
layer3=tf.maximum(0.01*layer3,layer3)
layer3=tf.nn.dropout(layer3, keep_prob_)
layer3=tf.layers.max_pooling2d(layer3, 2, 2,'same')

# layer4_1=tf.layers.conv2d(layer3, filters_*8, 2, 1, 'same',kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),name='layer4')
# layer4=tf.layers.batch_normalization(layer4_1)
# layer4=tf.maximum(0.01*layer4,layer4)
# layer4=tf.nn.dropout(layer4, keep_prob_)
# layer4=tf.layers.max_pooling2d(layer4, 2, 2,'same')

layer5=tf.reshape(layer3, [-1,2*2*filters_*4])

layer5=tf.layers.dense(layer5, 512)
layer5=tf.layers.batch_normalization(layer5)
layer5=tf.maximum(0.01*layer5,layer5)
layer5=tf.nn.dropout(layer5, keep_prob_)

layer6=tf.layers.dense(layer5, 512)
layer6=tf.layers.batch_normalization(layer6)
layer6=tf.maximum(0.01*layer6,layer6)
layer6=tf.nn.dropout(layer6, keep_prob_)

output=tf.layers.dense(layer6,10,tf.nn.softmax)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(output,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(12):
        sess.run(tf.assign(lr,(0.9**(i+1))*0.001))
        for j in range(n_batch):
#             player1_1,player1,player2_1,player2,player3_1,player3,player5=sess.run([layer1_1,layer1,layer2_1,layer2,layer3_1,layer3,layer5],{x:x_train[j*batch_size:(j+1)*batch_size],keep_prob_:0.5} )
#             print(player1_1.shape,player1.shape,player2_1.shape,player2.shape,player3_1.shape,player3.shape,player5.shape)
            sess.run(train_step,{x:x_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size],keep_prob_:0.5})
        acc1=sess.run(acc,{x:x_test,y:y_test,keep_prob_:1.0})
        print('第%s次训练准确率为%s'%(i,acc1))
