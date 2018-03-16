# -*- coding: utf-8 -*-
'''
Created on 2017年11月30日
@author: Administrator
'''
# import tensorflow as tf
# # from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib.pyplot as plt
from load_digist import sklearn_data
x,y=sklearn_data()
print(x.shape,y.shape)

# digits=load_digits()
# x=digits.data
# y=digits.target
# x_skl_train=x[:1000,64]
# y_skl_train=y[:1000]
# x_skl_test=x[1000:len(x),64]
# y_skl_test=y[1000:len(y)]
# 
# # minst=input_data.read_data_sets('Minst_data',one_hot=True)
# bacth_size=200
# n_bacth=len(x_skl_test)//bacth_size
# 
# # n_bacth=minst.train.num_examples//bacth_size
# 
# # 占位
# x=tf.placeholder(tf.float32,[None,64])
# y=tf.placeholder(tf.float32,[None,10])
# keep_drop=tf.placeholder(tf.float32)
# 
# layel_1=100
# layel_2=100
# layel_3=100
# # 隐藏层神经元数量
# 
# # 权重初始化
# w_1=tf.Variable(tf.truncated_normal([784,layel_1]))
# b_1=tf.Variable(tf.zeros([layel_1])+0.1)
# prediction_1=tf.nn.sigmoid(tf.matmul(x,w_1)+b_1)
# L1_dropout=tf.nn.dropout(prediction_1,keep_drop)
# 
# w_2=tf.Variable(tf.truncated_normal([layel_1,layel_2]))
# b_2=tf.Variable(tf.zeros([layel_2])+0.1)
# prediction_2=tf.nn.sigmoid(tf.matmul(L1_dropout,w_2)+b_2)
# L2_dropout=tf.nn.dropout(prediction_2,keep_drop)
# 
# w_3=tf.Variable(tf.truncated_normal([layel_2,layel_3]))
# b_3=tf.Variable(tf.zeros([layel_3])+0.1)
# prediction_3=tf.nn.tanh(tf.matmul(L2_dropout,w_3)+b_3)
# L3_dropout=tf.nn.dropout(prediction_3,keep_drop)
# 
# w_4=tf.Variable(tf.truncated_normal([layel_3,10]))
# b_4=tf.Variable(tf.zeros([10])+0.1)
# prediction=tf.nn.softmax(tf.matmul(L3_dropout,w_4)+b_4)
# 
# # loss1=tf.reduce_mean(tf.square(prediction2-prediction1))
# # train_step1=tf.train.GradientDescentOptimizer(0.01).minimize(loss1)
# # 二次代价函数，最小化loss
# # loss=tf.reduce_mean(tf.square(y-prediction1))
# 
# # 交叉熵函数
# loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 
# train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# # train_step=tf.train.AdadeltaOptimizer(0.5).minimize(loss)
# 
# # correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# # argmax(),返回一维张量中最大值所在的位置
# # 结果存放在一个布尔型列表中
# 
# # accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# # 求准确率
# 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     x_epoch,y_test_list,y_train_list=[],[],[]
#     for epoch in range(51):
#         for bacth_stpe in range(n_bacth):
#             sess.run(train_step,feed_dict={x:x_skl_train[bacth_size*bacth_stpe:bacth_size*(bacth_stpe+1),64],y:y_skl_train[bacth_size*bacth_stpe:bacth_size*(bacth_stpe+1)],keep_drop:0.9})
#             y_prediction=sess.run(prediction,feed_dict={x:x_skl_test,y:y_skl_test,keep_drop:1})
#         print('第%s次训练准确率:test:%s,train:%s'%(epoch,y_prediction))
# #     prediction_test=sess.run(prediction,feed_dict={x:x_test})
#       
# #     plt.figure()
# #     plt.plot(x_epoch,y_test_list,'r-',lw=2)
# #     plt.plot(x_epoch,y_train_list,'g-',lw=2)
# #     plt.show()    
# 
# print('done')
#         
#         
#         