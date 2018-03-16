# -*- coding: utf-8 -*-
'''
Created on 2018年1月10日
@author: Administrator
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
x_train,y_train=[],[]
for i in range(1,6):
    datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//data_batch_%s'%(i))
    for j in range(len(datadict[b'data'])):
        x_train.append(datadict[b'data'][j])
        y_train.append(datadict[b'labels'][j])
datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//test_batch')
x_test,y_test=datadict[b'data'],datadict[b'labels']
x_train,x_test,y_train1,y_test1=np.array(x_train)/127.5-1,np.array(x_test)/127.5-1,np.array(y_train),np.array(y_test)

y_train=np.zeros([50000,10])
for i in range(50000):
    y_train[i,y_train1[i]]=1 
y_test=np.zeros([10000,10])
for i in range(10000):
    y_test[i,y_test1[i]]=1          
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

filters_=64
batch_size=128
n_batch=50000//batch_size


x=tf.placeholder(tf.float32, [None,3072], name='input_data')
inputs_data=tf.reshape(x,[-1,32,32,3])
y=tf.placeholder(tf.float32,[None,10],name='lables')
keep_prob_=tf.placeholder(tf.float32)
lr=tf.Variable(0.0001,dtype=tf.float32)

layer1=tf.layers.conv2d(inputs_data, filters_, 3, 1,'same',name='layer1')
layer1=tf.layers.batch_normalization(layer1)
layer1=tf.maximum(0.01*layer1,layer1)
layer1=tf.nn.dropout(layer1, keep_prob_)
layer1=tf.layers.max_pooling2d(layer1, 2, 2,'same')

layer2=tf.layers.conv2d(layer1, filters_*2, 3, 1,'same',name='layer2')
layer2=tf.layers.batch_normalization(layer2)
layer2=tf.maximum(0.01*layer2,layer2)
layer2=tf.nn.dropout(layer2, keep_prob_)
layer2=tf.layers.max_pooling2d(layer2, 2, 2,'same')

layer3=tf.layers.conv2d(layer2, filters_*4, 3, 1,'same',name='layer3')
layer3=tf.layers.batch_normalization(layer3)
layer3=tf.maximum(0.01*layer3,layer3)
layer3=tf.nn.dropout(layer3, keep_prob_)
layer3=tf.layers.max_pooling2d(layer3, 2, 2,'same')

layer4=tf.layers.conv2d(layer3, filters_*8, 3, 1,'same',name='layer4')
layer4=tf.layers.batch_normalization(layer4)
layer4=tf.maximum(0.01*layer4,layer4)
layer4=tf.nn.dropout(layer4, keep_prob_)
layer4=tf.layers.max_pooling2d(layer4, 2, 2,'same')

layer5=tf.reshape(layer4, [-1,2*2*filters_*8])

layer5=tf.layers.dense(layer5, 2*2*filters_*8)
layer5=tf.layers.batch_normalization(layer5)
layer5=tf.maximum(0.01*layer5,layer5)
layer5=tf.nn.dropout(layer5, keep_prob_)

layer6=tf.layers.dense(layer5, 2*2*filters_*8)
layer6=tf.layers.batch_normalization(layer6)
layer6=tf.maximum(0.01*layer6,layer6)
layer6=tf.nn.dropout(layer6, keep_prob_)

output=tf.layers.dense(layer6,10,tf.nn.softmax)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(output,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
acc_list=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        sess.run(tf.assign(lr,(0.9**(i+1))*0.0001))
        for j in range(n_batch):
            sess.run(train_step,{x:x_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size],keep_prob_:0.5})
        acc1=sess.run(acc,{x:x_test[:batch_size],y:y_test[:batch_size],keep_prob_:1.0})
        acc_list.append(acc1)
        print('第%s次训练准确率为%s'%(i,acc1))

plt.plot(acc_list)
plt.show()










