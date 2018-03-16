# -*- coding: utf-8 -*-
'''
Created on 2017年12月13日
@author: Administrator
'''
# import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
# 
# import numpy as np

# minst=input_data.read_data_sets('Minst_data',one_hot='True')
# 
# x=minst.train.images[50]
# y=x.reshape(28,28)
# print(y)
# 
# plt.imshow(y,cmap='Greys_r')
# plt.show()
# 
# print('done')


# x=np.random.uniform(-1,1,[1000,100])
# print(x.shape)
# plt.plot(x[0])
# plt.show()

# y=np.random.randn(5,5)
# print(y)
# 
# from sklearn.preprocessing import MinMaxScaler
# minmax=MinMaxScaler()
# 
# y=minmax.fit_transform(y)
# print(y)

# x=np.random.random(2)
# y=np.random.randint(1,10)
# print(y)



# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

minst=input_data.read_data_sets('Minst_data',one_hot=True)

n_input=28
max_time=28
lstm_size=100
n_classes=10
batch_size=50
n_batch=minst.train._num_examples//batch_size

x=tf.placeholder(tf.float32, [None,784])
y=tf.placeholder(tf.float32,[None,10])
x_inputs=tf.reshape(x, [-1,28,28,1])

conv1=tf.layers.conv2d(inputs=x_inputs, filters=64, kernel_size=3, strides=1, padding='same',activation=tf.nn.relu)
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

prediction=tf.layers.dense(output2,10,tf.nn.softmax)


cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
acc_list=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
        for batch in range(1):
            batch_xs,batch_ys=minst.train.next_batch(batch_size)
            sess.run(train_step,{x:batch_xs,y:batch_ys})
        acc,xconv1,xpool1,xconv2,xpool2,xconv3,xpool3,xconv4,xpool4=sess.run([accuracy,conv1,pool1,conv2,pool2,conv3,pool3,conv4,pool4],{x:minst.test.images,y:minst.test.labels})
        acc_list.append(acc)
        print('第%s次，准确率：%s'%(epoch,acc))
        print(xconv1.shape,xpool1.shape,xconv2.shape,xpool2.shape,xconv3.shape,xpool3.shape,xconv4.shape,xpool4.shape)
        
plt.plot(acc)
plt.show()    
    
    
    
    
    
    
    
    
    
