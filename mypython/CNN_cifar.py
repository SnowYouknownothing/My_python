# -*- coding: utf-8 -*-
'''
Created on 2017年12月11日
@author: Administrator
'''
from numpy import array, reshape
import tensorflow as tf
import numpy as np
import time
# from matplotlib.pyplot import autumn

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
x_test,y_test=[],[]
for i in range(1,6):
    datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//data_batch_%s'%(i))
    for j in range(len(datadict[b'data'])):
        x_test.append(datadict[b'data'][j])
        y_test.append(datadict[b'labels'][j])
print(len(x_test),len(x_test[0]),len(y_test))

datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//test_batch')
x_train,y_train=datadict[b'data'],datadict[b'labels']
print(len(x_train),len(y_train))
x_test,y_test1,x_train,y_train1=array(x_test)/255,array(y_test),array(x_train)/255,array(y_train)

y_train=np.eye(10000,10)*0
for i in range(10000):
    y_train[i,y_train1[i]]=1          
y_test=np.eye(50000,10)*0
for i in range(50000):
    y_test[i,y_test1[i]]=1         

# x=reshape(x_test[66],[32,32,3])

# import pylab as pl
# import matplotlib.pyplot as plt
# 
# pl.gray()
# pl.matshow(x)
# pl.show()
# 
# 
# 
# plt.figure()
# plt.imshow(img)
# plt.show()


# # 分隔符
# 
import matplotlib.pyplot as plt
 
x=tf.placeholder(tf.float32,[None,3072])
y=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,32,32,3])
keep_drop=tf.placeholder(tf.float32)
lr=tf.Variable(1e-4,dtype=tf.float32)
  
  
# w_conv1=tf.Variable(tf.truncated_normal([3,3,3,16],stddev=0.1))
# b_conv1=tf.Variable(tf.constant(0.1,shape=[16]))
# h_conv1=tf.nn.relu(tf.nn.conv2d(x_image,w_conv1, strides=[1,1,1,1], padding='SAME')+b_conv1)
# h_pool1=tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
h_conv1=tf.layers.conv2d(
    inputs=x_image,
    filters=64,
    kernel_size=[5,5],
    strides=1,
    padding='SAME',
    activation=tf.nn.relu)
h_pool1=tf.layers.max_pooling2d(
    inputs=h_conv1, 
    pool_size=[2,2], 
    strides=[2,2], 
    padding='SAME')
h_conv2=tf.layers.conv2d(
    inputs=h_pool1,
    filters=64,
    kernel_size=[5,5],
    strides=1,
    padding='SAME',
    activation=tf.nn.relu)
h_pool2=tf.layers.average_pooling2d(
    inputs=h_conv2,
    pool_size=[2,2],
    strides=[2,2],
    padding='SAME')
 
h_conv3=tf.layers.conv2d(
    inputs=h_pool2,
    filters=128,
    kernel_size=[5,5],
    strides=1,
    padding='SAME',
    activation=tf.nn.relu)
h_pool3=tf.layers.average_pooling2d(
    inputs=h_conv3,
    pool_size=[2,2],
    strides=[2,2],
    padding='SAME')

h_conv4=tf.layers.conv2d(
    inputs=h_pool3,
    filters=128,
    kernel_size=[3,3],
    strides=1,
    padding='SAME',
    activation=tf.nn.relu)
h_pool4=tf.layers.max_pooling2d(
    inputs=h_conv4,
    pool_size=[2,2],
    strides=[2,2],
    padding='SAME')
 
h_pool4_flat=tf.reshape(h_pool4,[-1,2*2*128])
 
# w_conv2=tf.Variable(tf.truncated_normal([3,3,16,32],stddev=0.1))
# b_conv2=tf.Variable(tf.constant(0.1,shape=[32]))
# h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,w_conv2, strides=[1,1,1,1], padding='SAME')+b_conv2)
# h_pool2=tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# # h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*60])
# 
# w_conv3=tf.Variable(tf.truncated_normal([3,3,32,64],stddev=0.1))
# b_conv3=tf.Variable(tf.constant(0.1,shape=[64]))
# h_conv3=tf.nn.relu(tf.nn.conv2d(h_pool2,w_conv3, strides=[1,1,1,1], padding='SAME')+b_conv3)
# h_pool3=tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# # h_pool3_flat=tf.reshape(h_pool3,[-1,4*4*60])
# 
# w_conv4=tf.Variable(tf.truncated_normal([2,2,64,128],stddev=0.1))
# b_conv4=tf.Variable(tf.constant(0.1,shape=[128]))
# h_conv4=tf.nn.relu(tf.nn.conv2d(h_pool3,w_conv4, strides=[1,1,1,1], padding='SAME')+b_conv4)
# h_pool4=tf.nn.max_pool(h_conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# h_pool4_flat=tf.reshape(h_pool4,[-1,2*2*128])
 
# w_1=tf.Variable(tf.truncated_normal([2*2*128,512],stddev=0.1))
# b_1=tf.Variable(tf.constant(0.1,shape=[512]))
# fc_1=tf.nn.relu(tf.matmul(h_pool4_flat,w_1)+b_1)
 
dense1=tf.layers.dense(h_pool4_flat,512,tf.nn.relu)
fc_1_dropout=tf.nn.dropout(dense1, keep_drop)
 
# w_2=tf.Variable(tf.truncated_normal([512,512],stddev=0.1))
# b_2=tf.Variable(tf.constant(0.1,shape=[512]))
# fc_2=tf.nn.relu(tf.matmul(fc_1_dropout,w_2)+b_2)
 
dense2=tf.layers.dense(fc_1_dropout,512,tf.nn.relu)
fc_2_dropout=tf.nn.dropout(dense2, keep_drop)
 
# w_3=tf.Variable(tf.truncated_normal([512,10],stddev=0.1))
# b_3=tf.Variable(tf.constant(0.1,shape=[10]))
# prediction=tf.nn.softmax(tf.matmul(fc_2_dropout,w_3)+b_3)
 
dense3=tf.layers.dense(fc_2_dropout,512,tf.nn.relu)
fc_3_dropout=tf.nn.dropout(dense3, keep_drop)
 
             
prediction=tf.layers.dense(fc_2_dropout,10,tf.nn.softmax)                 
                 
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step=tf.train.AdamOptimizer(lr).minimize(loss)
   
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# argmax(),返回一维张量中最大值所在的位置
# 结果存放在一个布尔型列表中
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 
start1=time.time()
acc_list=[]
saver=tf.train.Saver()
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        sess.run(tf.assign(lr,0.001*(0.99**i)))
        for epoch in range(100):
            sess.run(train_step,feed_dict={x:x_test[epoch*100:(epoch+1)*100],y:y_test[epoch*100:(epoch+1)*100],keep_drop:0.5})
#             print(epoch)
        acc=sess.run(accuracy,feed_dict={x:x_train[:500],y:y_train[:500],keep_drop:1.0})
        print(i,acc)
        acc_list.append([acc])
     
    saver.save(sess,'net/my_net.ckpt')
     
end1=time.time()
 
print(acc_list) 
plt.figure()
plt.plot(acc_list,'r-',lw=2)
plt.show()
print((end1-start1)/3600)
      
print('done')












