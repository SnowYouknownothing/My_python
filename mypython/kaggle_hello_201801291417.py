# -*- coding: utf-8 -*-
'''
Created on 2018年1月29日
@author: Administrator
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def load_data(path):      
    with open(path,encoding='utf-8') as fn:
        reader=csv.reader(fn)
        rows=[row for row in reader]
    rows.pop(0)
    x_train=[]
    for i in range(len(rows)):
        c=rows[i]
        x_1=[]
        for j in range(len(rows[0])):
            x_1.append(int(c[j]))
        x_train.append(x_1)
    x_train=np.array(x_train)
    print(x_train.shape)
    if len(x_train)<30000:
        return x_train/255
    else:
        x_images=x_train[:,1:]
        x_labels=x_train[:,0]
        print(x_images.shape,x_labels.shape)
        print(x_images.shape)
        print(np.max(x_images),np.min(x_images))
        x_labels_=np.zeros([len(x_labels),10])
        for i in range(len(x_labels)):
            x_labels_[i,x_labels[i]]=1
    #     print(x_labels[1:5],x_labels_[1:5])
        x_images=x_images/255
        return x_images,x_labels_

'''
x_images=np.reshape(x_images,[len(x_images),28,28])
print(x_labels[:200])
fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=x_images[:200]
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()
'''
def train_(x_train,x_test,y_train,y_test,x_prediction):
    x=tf.placeholder(tf.float32,[None,784])
    inputs_data=tf.reshape(x, [-1,28,28,1])
    y=tf.placeholder(tf.float32,[None,10])
    keep_prob_=tf.placeholder(tf.float32)
    lr=tf.Variable(0.0001,dtype=tf.float32)

    filters_=64
    batch_size=64
    n_batch=len(x_train)//batch_size
    
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
    
    layer3=tf.layers.conv2d(layer2, filters_*4, 4, 1,'valid',name='layer3')
    layer3=tf.layers.batch_normalization(layer3)
    layer3=tf.maximum(0.01*layer3,layer3)
    layer3=tf.nn.dropout(layer3, keep_prob_)
    layer3_=tf.layers.max_pooling2d(layer3, 2, 2,'same')
    
#     layer4=tf.layers.conv2d(layer3, filters_*8, 3, 1,'same',name='layer4')
#     layer4=tf.layers.batch_normalization(layer4)
#     layer4=tf.maximum(0.01*layer4,layer4)
#     layer4=tf.nn.dropout(layer4, keep_prob_)
#     layer4=tf.layers.max_pooling2d(layer4, 2, 2,'same')
    
    layer5=tf.reshape(layer3_, [-1,2*2*filters_*4])
    
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
    train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
      
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(output,1))
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    acc_list=[]
    
    prediction=tf.argmax(output,1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(31):
            if i % 10 ==0:
                sess.run(tf.assign(lr,(10**(-i/10))*0.0001))
            for j in range(n_batch):
                sess.run(train_step,{x:x_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size],keep_prob_:0.5})
#                 a,b,c,d=sess.run([layer1,layer2,layer3,layer3_],{x:x_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size],keep_prob_:0.5})
#                 print(a.shape,b.shape,c.shape,d.shape)
            acc1=sess.run(acc,{x:x_test,y:y_test,keep_prob_:1.0})
            acc_list.append(acc1)
            print('第%s次训练准确率为%s'%(i,acc1))
        prediction_out=sess.run(prediction,{x:x_prediction,keep_prob_:1.0})
    
    return acc_list,prediction_out


name_list=['train.csv','test.csv','sample_submission.csv']
path_list=[]
for i in range(3):
    path='D:\\Desktop\\kaggle\\digits_20180122\\'+name_list[i]
    path_list.append(path)
print(path_list)
path_train=path_list[0]
path_test=path_list[1]

# minst=input_data.read_data_sets('minst_data',one_hot=True)
# x_train_,y_train_=minst.train.images,minst.train.labels
# x_test_,y_test_=minst.test.images,minst.test.labels
# print(x_train_.shape,y_train_.shape,x_test_.shape,y_test_.shape)
# x_sum_train=np.concatenate((x_train_,x_test_),axis=0)
# y_sum_train=np.concatenate((y_train_,y_test_),axis=0)

# def more_data(x=None):
#     
# 
#     pass
# 
#     return 

if __name__=='__main__':
    x_train_,y_train_=load_data(path_train)
    x_sum_test=load_data(path_test)
    more_data1=np.load('D:\\Desktop\\kaggle\\digits_20180122\\more_data1_py\\new_data1.npy')
    more_data2=np.load('D:\\Desktop\\kaggle\\digits_20180122\\more_data1_py\\new_data2.npy')
#     more_data3=np.load('D:\\Desktop\\kaggle\\digits_20180122\\more_data_py\\new_data1.npy')
#     more_data4=np.load('D:\\Desktop\\kaggle\\digits_20180122\\more_data_py\\new_data2.npy')    

#     x_train_=np.concatenate((x_train_,more_data1),axis=0)
#     x_train_=np.concatenate((x_train_,more_data2),axis=0)
# #     x_train_=np.concatenate((x_train_,more_data3),axis=0)
# #     x_train_=np.concatenate((x_train_,more_data4),axis=0)
#     y_train_1=np.concatenate((y_train_,y_train_),axis=0)
# #     y_train_2=np.concatenate((y_train_1,y_train_),axis=0)
# #     y_train_3=np.concatenate((y_train_2,y_train_),axis=0)    
#     y_train_=np.concatenate((y_train_1,y_train_),axis=0)
  
    print(x_train_.shape,y_train_.shape)
#     x_sum_train=np.concatenate((x_sum_train,x_train_),axis=0)
#     y_sum_train=np.concatenate((y_sum_train,y_train_),axis=0)
#     print(x_sum_train.shape,y_sum_train.shape) 
#     print(np.max(x_sum_train),np.min(x_sum_train),np.max(y_sum_train),np.min(y_sum_train))
#     x_images=np.reshape(x_sum_train,[len(x_sum_train),28,28])
#     fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
#     imgs=x_images[10000:10200]
#     for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
#         for img,ax in zip(image,row):
#             ax.imshow(img)
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#     fig.tight_layout(pad=0.1)
#     plt.show()
# 
#     imgs=x_images[10200:10400]
#     for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
#         for img,ax in zip(image,row):
#             ax.imshow(img)
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#     fig.tight_layout(pad=0.1)
#     plt.show()

    x_train,x_test,y_train,y_test=train_test_split(x_train_,y_train_,test_size=0.1)
    acc_list,prediction_output=train_(x_train,x_test,y_train,y_test,x_prediction=x_sum_test)
    plt.plot(acc_list)
    plt.show()
    
with open('D:\\Desktop\\kaggle\\digits_20180122\\kaggle_test_201802021707.csv','w',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['Imageld','Label'])
    for i in range(len(prediction_output)):
        data=(i+1,prediction_output[i])
        writer.writerow(data)


print('done!')