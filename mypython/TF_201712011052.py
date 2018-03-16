# -*- coding: utf-8 -*-
'''
Created on 2017年12月1日
@author: Administrator
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
minst=input_data.read_data_sets('Minst_data',one_hot=True)
bacth_size=100
n_bacth=minst.train.num_examples//bacth_size

layel_1=500
layel_2=1000
layel_3=1000
lr=tf.Variable(0.01,dtype=tf.float32)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
#         平均值
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
#         标准差
        tf.summary.scalar('max',tf.reduce_max(var))
        
        tf.summary.scalar('min',tf.reduce_min(var))
        
        tf.summary.histogram('histogram',var)
#         直方图


with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x_input')
    y=tf.placeholder(tf.float32,[None,10],name='y_input')
# 隐藏层神经元数量
# 权重初始化
with tf.name_scope('laye1'):
    with tf.name_scope('weight1'):
        w1=tf.Variable(tf.truncated_normal([784,layel_1]),name='w1')
        variable_summaries(w1)
        
    with tf.name_scope('biases1'):
        b1=tf.Variable(tf.zeros([layel_1]),name='b1')
        variable_summaries(b1)
        
    with tf.name_scope('L_j1'):
        L_j1=tf.matmul(x,w1)+b1
#         tf.summary.scalar('L_j',L_j)
    with tf.name_scope('softmax1'):
        prediction1=tf.nn.tanh(L_j1) 


with tf.name_scope('laye2'):
    with tf.name_scope('weight'):
        w=tf.Variable(tf.truncated_normal([layel_1,10]),name='w')
        variable_summaries(w)
    with tf.name_scope('biases'):
        b=tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('L_j'):
        L_j=tf.matmul(prediction1,w)+b
#         tf.summary.scalar('L_j',L_j)
    with tf.name_scope('softmax'):
        prediction=tf.nn.softmax(L_j) 
      
#         tf.summary.scalar('prediction',prediction)
# 二次代价函数，最小化loss
# loss=tf.reduce_mean(tf.square(y-prediction))

# 交叉熵函数
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train_step'):
    train_step=tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope('acc_sum'):   
    with tf.name_scope('correct'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # argmax(),返回一维张量中最大值所在的位置
    # 结果存放在一个布尔型列表中
    with tf.name_scope('acc'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('acc',accuracy)
# 求准确率
merged=tf.summary.merge_all()
# 合并所有的summary

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('logs',sess.graph)
    for epoch in range(51):
        sess.run(tf.assign(lr,0.01*(0.95**epoch)))
        for bacth_stpe in range(n_bacth):
            x_data,y_data=minst.train.next_batch(bacth_size)
            summary,_=sess.run([merged,train_step],feed_dict={x:x_data,y:y_data})
#             sess.run(train_step1)
        writer.add_summary(summary,epoch)
        x_test,y_test=minst.test.images,minst.test.labels
        x_train,y_train=minst.train.images,minst.train.labels
        test_acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
        train_acc=sess.run(accuracy,feed_dict={x:x_train,y:y_train})
#         if epoch%5==0:
#         print('第%s次训练准确率:test:%s,train:%s,lr:%s'%(epoch,test_acc,train_acc,sess.run(lr)))
print('done',test_acc,train_acc)
               
        