# -*- coding: utf-8 -*-
'''
Created on 2018年1月24日
@author: Administrator
'''

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data

#inception-V3瓶颈层节点个数
BOTTLENECK_TENSOR_SIZE = 2048
#瓶颈层tenbsor name
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
#图像输入tensor name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
# v3 path
MODEL_DIR = 'D:\\Desktop\\kaggle\\inception_V3_eg\\inception_model'
# v3 modefile
MODEL_FILE= 'classify_image_graph_def.pb'

N_CLASSES=10

N_BATCH=100
batch_size=128



minst=input_data.read_data_sets('minst_data',one_hot=True)
x_train_,y_train_=minst.train.images,minst.train.labels
x_test_,y_test_=minst.test.images,minst.test.labels
print(x_train_.shape,y_train_.shape,x_test_.shape,y_test_.shape)


x_input1=tf.placeholder(tf.float32,[None,784])
x_input2=tf.reshape(x_input1,[-1,28,28,1])

with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE),'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])



x_middle=tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE])
y_output=tf.placeholder(tf.float32,[None,N_CLASSES])

weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, N_CLASSES], stddev=0.001))
biases = tf.Variable(tf.zeros([N_CLASSES]))
logits = tf.matmul(x_middle, weights) + biases
final_tensor = tf.nn.softmax(logits)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_output, logits=logits))
train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
  
correct_prediction=tf.equal(tf.argmax(y_output,1),tf.argmax(final_tensor,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(10):
        for i in range(N_BATCH):
            x_1=sess.run(x_input2,{x_input1:x_train_[i*batch_size:(i+1)*batch_size]})
            x_m=sess.run(bottleneck_tensor,{jpeg_data_tensor:x_1})  
            x_m=np.squeeze(x_m)
            sess.run(train_step,{x_middle:x_m,y_output:x_train_[i*batch_size:(i+1)*batch_size]})
        x_2=sess.run(x_input2,{x_input1:x_test_[i*batch_size:(i+1)*batch_size]})
        x_n=sess.run(bottleneck_tensor,{jpeg_data_tensor:x_2})  
        acc=sess.run(acc,{x_middle:x_n,y_output:x_test_[i*batch_size:(i+1)*batch_size]})
        print(j,acc)
            
        
        
        
        
        
