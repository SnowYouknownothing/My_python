# -*- coding: utf-8 -*-
'''
Created on 2018年2月6日
@author: Administrator
'''
import csv
import numpy as np
from sklearn import svm


def get_data(path):
    with open(path) as fn:
        readers=csv.reader(fn)
        rows=[row for row in readers]
    rows.pop(0) 
    return np.array(rows)
        
path_train='D:\\Desktop\\kaggle\\Titanic_20180202\\train.csv'
path_test='D:\\Desktop\\kaggle\\Titanic_20180202\\test.csv'

train_data=get_data(path_train)
test_data=get_data(path_test)
print(train_data.shape,test_data.shape)
# print(test_data[1],train_data[1])
# for i in range(len(train_data)):
#     train_data.remove(i,3)
    
print(train_data.shape,test_data.shape)   

x_zero=np.zeros((len(test_data),1))
test_data=np.hstack((x_zero,test_data))

def trans(train_data):

    x_train_1=np.reshape(train_data[:,2],[int(len(train_data)),1])
    x_train_2=train_data[:,4:8]
    x_train_3=np.reshape(train_data[:,-3],[int(len(train_data)),1])
    x_train_4=np.reshape(train_data[:,-1],[int(len(train_data)),1])
    
    x_train=np.hstack((x_train_1,x_train_2))
    x_train=np.hstack((x_train,x_train_3))
    x_train=np.hstack((x_train,x_train_4))
    
    y_train=np.reshape(train_data[:,1],[int(len(train_data)),1])

    for i in range(len(x_train)):
        if x_train[i,1]=='female':
            x_train[i,1]=0
        else:
            x_train[i,1]=1
        
        if x_train[i,-1]=='C':
            x_train[i,-1]=0
        if x_train[i,-1]=='Q':
            x_train[i,-1]=1
        if x_train[i,-1]=='S':
            x_train[i,-1]=2
        if x_train[i,2]=='':
            x_train[i,2]=np.random.randint(10,60)
    return x_train ,y_train

x_train,y_train=trans(train_data)
x_test,_=trans(test_data)
print(x_train.shape,y_train.shape)
print(x_test.shape)
print(x_train[5],y_train[5])
print(x_test[5])

x_train_=[]
y_train_=np.zeros([len(y_train),1])
for i in range(len(x_train)):
    if str(y_train[i])=="['0']":
        y_train_[i]=0
    else:
        y_train_[i]=1
    c=x_train[i]
    x_1=[]
    for j in range(len(x_train[0])):
        if '.' in c[j]:
            c[j]=c[j][:(c[j].index('.'))]
        if c[j]=='':
            c[j]=0
        x_1.append(int(c[j]))
    x_train_.append(x_1) 
x_train_=np.array(x_train_)

x_test_=[]

for i in range(len(x_test)):
    c=x_test[i]
    x_1=[]
    for j in range(len(x_test[0])):
        if '.' in c[j]:
            c[j]=c[j][:(c[j].index('.'))]
        if c[j]=='':
            c[j]=0
        x_1.append(int(c[j]))
    x_test_.append(x_1) 
    
x_test=np.array(x_test_)


print(x_train_.shape,x_train_[5])
print(y_train_.shape,y_train_[5])
print(x_test.shape,x_test[5])

y_train__=np.zeros((len(y_train_),2))
print(y_train_[:5])
for i in range(len(y_train_)):
    y_train__[i,int(y_train_[i]) ]=1



y_train_=y_train__

print(y_train_[:5])

import tensorflow as tf

x=tf.placeholder(tf.float32, [None,7])
y=tf.placeholder(tf.float32, [None,2])
keep_prob_=tf.placeholder(tf.float32)


# layer1=tf.layers.dense(x, 1024, activation=tf.nn.relu)
# layer1=tf.nn.dropout(layer1, keep_drop)
# 
# layer2=tf.layers.dense(layer1, 1024, activation=tf.nn.relu)
# layer2=tf.nn.dropout(layer2, keep_drop)
# 
# layer3=tf.layers.dense(layer2, 1024, activation=tf.nn.relu)
# layer3=tf.nn.dropout(layer3, keep_drop)
# 
# layer4=tf.layers.dense(layer3, 1024, activation=tf.nn.relu)
# layer4=tf.nn.dropout(layer4, keep_drop)
filters_=32

layer5=tf.layers.dense(x, 2*2*filters_*8)
layer5=tf.layers.batch_normalization(layer5)
layer5=tf.maximum(0.01*layer5,layer5)
layer5=tf.nn.dropout(layer5, keep_prob_)

layer6=tf.layers.dense(layer5, 2*2*filters_*8)
layer6=tf.layers.batch_normalization(layer6)
layer6=tf.maximum(0.01*layer6,layer6)
layer6=tf.nn.dropout(layer6, keep_prob_)

layer7=tf.layers.dense(layer6, 2*2*filters_*8)
layer7=tf.layers.batch_normalization(layer7)
layer7=tf.maximum(0.01*layer7,layer7)
layer7=tf.nn.dropout(layer7, keep_prob_)

layer8=tf.layers.dense(layer7, 2*2*filters_*8)
layer8=tf.layers.batch_normalization(layer8)
layer8=tf.maximum(0.01*layer8,layer8)
layer8=tf.nn.dropout(layer8, keep_prob_)

layer9=tf.layers.dense(layer8, 2*2*filters_*8)
layer9=tf.layers.batch_normalization(layer9)
layer9=tf.maximum(0.01*layer9,layer9)
layer9=tf.nn.dropout(layer9, keep_prob_)
# prediction=tf.layers.dense(layer8, 2, activation=tf.nn.sigmoid)
# loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction))

prediction=tf.layers.dense(layer9, 2, activation=tf.nn.softmax)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

pre=tf.argmax(prediction,1)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# argmax(),返回一维张量中最大值所在的位置
# 结果存放在一个布尔型列表中
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        for i in range(25):
            sess.run(train_step,{x:x_train_[i*32:(i+1)*32],y:y_train_[i*32:(i+1)*32],keep_prob_:0.5})
#         prediction_new=sess.run(prediction,{x:x_train_[800:],keep_prob_:1})
        if epoch % 50==0:
            acc=sess.run(accuracy,{x:x_train_[800:],y:y_train_[800:],keep_prob_:1})
            print(acc)

print('done!')

#             acc=y_train_[800:] - np.round(prediction_new)
#             sum=0
#             for i in range(len(acc)):       
#                 if acc[i]<0.01:
#                     sum+=1
#             print(sum/(len(acc)))
    

