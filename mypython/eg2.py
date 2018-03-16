# -*- coding: utf-8 -*-
'''
Created on 2017年12月7日
@author: Administrator
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
minst=input_data.read_data_sets('Minst_data',one_hot=True)
bacth_size=100
n_bacth=minst.train.num_examples//bacth_size

# 占位
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_drop=tf.placeholder(tf.float32)
lr=tf.Variable(0.2,dtype=tf.float32)

layel_1=500
layel_2=500
layel_3=500
# 隐藏层神经元数量

# 权重初始化
w_1=tf.Variable(tf.truncated_normal([784,layel_1]))
b_1=tf.Variable(tf.zeros([layel_1]))
prediction_1=tf.nn.tanh(tf.matmul(x,w_1)+b_1)
L1_dropout=tf.nn.dropout(prediction_1,keep_drop)

w_2=tf.Variable(tf.truncated_normal([layel_1,layel_2]))
b_2=tf.Variable(tf.zeros([layel_2])+0.1)
prediction_2=tf.nn.tanh(tf.matmul(L1_dropout,w_2)+b_2)
L2_dropout=tf.nn.dropout(prediction_2,keep_drop)
 
w_3=tf.Variable(tf.truncated_normal([layel_2,layel_3]))
b_3=tf.Variable(tf.zeros([layel_3])+0.1)
prediction_3=tf.nn.tanh(tf.matmul(L2_dropout,w_3)+b_3)
L3_dropout=tf.nn.dropout(prediction_3,keep_drop)

# w_4=tf.Variable(tf.truncated_normal([layel_3,10]))
# b_4=tf.Variable(tf.zeros([10])+0.1)
# prediction=tf.nn.softmax(tf.matmul(L3_dropout,w_4)+b_4)

w=tf.Variable(tf.truncated_normal([layel_3,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(L3_dropout,w)+b)

# 二次代价函数，最小化loss
# loss=tf.reduce_mean(tf.square(y-prediction))

# 交叉熵函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# train_step=tf.train.AdadeltaOptimizer(0.6).minimize(loss)
# train_step=tf.train.AdamOptimizer(lr).minimize(loss)


correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# argmax(),返回一维张量中最大值所在的位置
# 结果存放在一个布尔型列表中

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 求准确率
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_epoch,y_test_list,y_train_list=[],[],[]
    for epoch in range(51):
        sess.run(tf.assign(lr,0.2*(0.95**epoch)))
        for bacth_stpe in range(n_bacth):
            x_data,y_data=minst.train.next_batch(bacth_size)
            sess.run(train_step,feed_dict={x:x_data,y:y_data,keep_drop:0.8})
#             sess.run(train_step1)
        x_test,y_test=minst.test.images,minst.test.labels
        x_train,y_train=minst.train.images,minst.train.labels
        test_acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test,keep_drop:1.0})
        train_acc=sess.run(accuracy,feed_dict={x:x_train,y:y_train,keep_drop:1.0})
#         if epoch%5==0:
        print('第%s次训练准确率:test:%s,train:%s,lr:%s'%(epoch,test_acc,train_acc,sess.run(lr)))
        x_epoch.append(epoch)
        y_test_list.append(test_acc)
        y_train_list.append(train_acc)
#     prediction_test=sess.run(prediction,feed_dict={x:x_test})
      
    plt.figure()
    plt.plot(x_epoch,y_test_list,'r-',lw=2)
    plt.plot(x_epoch,y_train_list,'g-',lw=1)
    plt.show()    

print('done',y_test_list[-1],y_train_list[-1])           