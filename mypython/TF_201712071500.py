# -*- coding: utf-8 -*-
'''
Created on 2017年12月8日
@author: Administrator
'''
import tensorflow as tf
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

x_data=(load_digits().data)/16
x_train=x_data[:1400,:64]
x_test=x_data[1400:len(x_data),:64]

y_data=(load_digits().target)
y_train1=y_data[:1400]
y_test1=y_data[1400:len(y_data)]

y_train=np.eye(1400,10)*0
for i in range(1400):
    y_train[i,y_train1[i]]=1          
y_test=np.eye(397,10)*0
for i in range(397):
    y_test[i,y_test1[i]-1]=1         
 
x=tf.placeholder(tf.float32,[None,64])
y=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,8,8,1])
keep_drop=tf.placeholder(tf.float32)
lr=tf.Variable(0.0001,dtype=tf.float32)
 
w_conv1=tf.Variable(tf.truncated_normal([4,4,1,16],stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[16]))
h_conv1=tf.nn.relu(tf.nn.conv2d(x_image,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
 
w_conv2=tf.Variable(tf.truncated_normal([4,4,16,32],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[32]))
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
h_pool2_flat=tf.reshape(h_pool2,[-1,2*2*32])
 
w_1=tf.Variable(tf.truncated_normal([2*2*32,512]))
b_1=tf.Variable(tf.constant(0.1,shape=[512])) 
prediction_0=tf.nn.relu(tf.matmul(h_pool2_flat,w_1)+b_1)
prediction_1=tf.nn.dropout(prediction_0,keep_drop)
 
w=tf.Variable(tf.truncated_normal([512,10]))
b=tf.Variable(tf.constant(0.1,shape=[10]))
prediction=tf.nn.softmax(tf.matmul(prediction_1,w)+b)
 
 
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
 
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# argmax(),返回一维张量中最大值所在的位置
# 结果存放在一个布尔型列表中
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 
acc_list,acc_train_list=[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        sess.run(tf.assign(lr,0.0001*(0.99**epoch)))
        for i in range(70):
            sess.run(train_step,feed_dict={x:x_train[i*20:(i+1)*20],y:y_train[i*20:(i+1)*20],keep_drop:0.7})
        acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test,keep_drop:1.0})
        acc_list.append(acc)
        acc_train=sess.run(accuracy,feed_dict={x:x_train,y:y_train,keep_drop:1.0})
        acc_train_list.append(acc_train)
#     print(acc)
#     for i in range(10):
#         print(y_prediction[i],y_test[10+i])
#     acc.append(y_test[epoch]-y_prediction)
plt.figure()
plt.plot(acc_list,'r-',lw=2)
plt.plot(acc_train_list,'g-',lw=1)
plt.show()
print('done')



























# print(digits.data)
# print(digits.target)
# print(digits.data.shape)
# print(digits.target.shape)
# plt.figure()
# plt.plot(x[0])
# plt.show()
# x_imgae=np.eye(8,8)
# print(x_imgae.shape)
# j=0
# for i in range(len(x[22])):
#     x_imgae[j,i%8]=x[22,i]
#     if (i+1)%8==0:
#         j+=1
# print(y[22])
# print(digits.images[4])
# print(x_imgae)
# plt.gray()
# plt.matshow(x_imgae)
# plt.show()