# -*- coding: utf-8 -*-
'''
Created on 2017年12月26日
@author: Administrator
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

minst=input_data.read_data_sets('minst_data',one_hot=True)
x_train=minst.train.images
print(x_train.shape)

n_input=28
max_time=28
lstm_size=100
n_classes=10
batch_size=50
n_batch=minst.train._num_examples//batch_size

x=tf.placeholder(tf.float32, [None,784])
y=tf.placeholder(tf.float32,[None,10])

weights=tf.Variable(tf.truncated_normal(shape=[lstm_size,n_classes], stddev=0.1))
biases=tf.Variable(tf.constant(0.1,shape=[n_classes]))

def CNN(x):
    inputs=tf.reshape(x, [-1,max_time,n_input,1])
    conv1=tf.layers.conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu)
    pool1=tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same')
    layer1=tf.nn.dropout(pool1, keep_prob=0.5)
    
    conv2=tf.layers.conv2d(layer1, filters=128, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu)
    pool2=tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='same')
    layer2=tf.nn.dropout(pool2, keep_prob=0.5)

    conv3=tf.layers.conv2d(layer2, filters=256, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu)
    pool3=tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, padding='valid')
    layer3=tf.nn.dropout(pool3, keep_prob=0.5)   

    conv4=tf.layers.conv2d(layer3, filters=512, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu)
    pool4=tf.layers.max_pooling2d(conv4, pool_size=2, strides=2, padding='same')
    layer4=tf.nn.dropout(pool4, keep_prob=0.5)
    
    full_conect=tf.reshape(layer4,[-1,2*2*512])
    output1=tf.layers.dense(full_conect, 512, activation=tf.nn.relu)
    output2=tf.nn.dropout(output1,0.5)
    
    output=tf.layers.dense(output2,10,tf.nn.softmax)
    
    return output


def RNN(x,weights,biases):
    inputs=tf.reshape(x,[-1,max_time,n_input])
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
#     final_state[0]:cell state
#     final_state[1]:hidden state
#     outputs[-1]=final_state[1]
    results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results

prediction_RNN=RNN(x, weights, biases)
cross_entropy_RNN=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction_RNN))
train_step_RNN=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_RNN)
correct_prediction_RNN=tf.equal(tf.argmax(y,1),tf.argmax(prediction_RNN,1))
accuracy_RNN=tf.reduce_mean(tf.cast(correct_prediction_RNN,tf.float32))

prediction_CNN=CNN(x)
cross_entropy_CNN=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction_CNN))
train_step_CNN=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_CNN)
correct_prediction_CNN=tf.equal(tf.argmax(y,1),tf.argmax(prediction_CNN,1))
accuracy_CNN=tf.reduce_mean(tf.cast(correct_prediction_CNN,tf.float32))

acc_RNN_list,acc_CNN_list=[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs,batch_ys=minst.train.next_batch(batch_size)
            sess.run([train_step_RNN,train_step_CNN],{x:batch_xs,y:batch_ys})
        acc_RNN=sess.run(accuracy_RNN,{x:minst.test.images,y:minst.test.labels})
        acc_CNN=sess.run(accuracy_CNN,{x:minst.test.images,y:minst.test.labels})
        acc_RNN_list.append(acc_RNN)
        acc_CNN_list.append(acc_CNN)
        print('第%s次，RNN准确率：%s，CNN准确率：%s'%(epoch,acc_RNN,acc_CNN))
        
plt.plot(acc_RNN_list,'red')
plt.plot(acc_CNN_list,'green')
plt.show()    
    
    
    
    
    
    
    
    
    