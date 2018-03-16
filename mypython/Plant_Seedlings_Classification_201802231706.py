# -*- coding: utf-8 -*-
'''
Created on 2018年2月26日
@author: Administrator
'''
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

new_width,new_high=128,128
train_data_path='D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\train_2\\train_2\\train'
test_data_path='D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\train_2\\test_2\\test'

#name_list=['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']
train_name_list_sum=os.listdir(train_data_path)
test_name_list=os.listdir(test_data_path)
train_name_list=[]
for i in range(len(train_name_list_sum)):
    train_name_list.append(os.listdir(train_data_path+'\\'+train_name_list_sum[i]))
    
print(np.array(train_name_list).shape)
print(np.array(train_name_list_sum).shape)
print(np.array(test_name_list).shape)
print(np.array(train_name_list[0]).shape)


trans_test_data=True
while trans_test_data==False:
    path=test_data_path+'\\'
    for file in os.listdir(path):
        im =Image.open(path+file)
        out=im.resize((new_width,new_high))
        out.save(path+file)
    trans_test_data=True

print('step1') 
load_test_data=True  
while load_test_data==False:
    path=test_data_path+'\\'
    faces_kt=[]
    for file in os.listdir(path):
        images=mpimg.imread(path+file)
        images=images[:,:,:3]
        faces_kt.append(images)
    np.save('Plant_Seedlings_Classification\\plant_new_data_test_2',np.array(faces_kt))
    print(np.array(faces_kt).shape)                  
    print('step2')
    load_test_data=True
x_test_sum=np.load('Plant_Seedlings_Classification\\plant_new_data_test_2.npy')
print(x_test_sum.shape)


trans_data=True
while trans_data==False:
    for i in range(len(train_name_list_sum)):
        faces_kt=[]
        path=train_data_path+'\\'+train_name_list_sum[i]+'\\'
        for file in os.listdir(path):
            im =Image.open(path+file)
            out=im.resize((new_width,new_high))
            out.save(path+file)
    trans_data=True
print('step3') 
load_data=True 
while load_data==False:
#     x_train=[]
#     faces_kt=[]   
    for i in range(len(train_name_list_sum)):
        path=train_data_path+'\\'+train_name_list_sum[i]+'\\'
        print(path)
#         j=0
        faces_kt=[]
        for file in os.listdir(path):
#             print(file)
#             print(path+file)
            images=mpimg.imread(path+file)
            images=images[:,:,:3]
            faces_kt.append(images)
        np.save('Plant_Seedlings_Classification\\plant_new_data_3'+str(i),np.array(faces_kt))
        print(np.array(faces_kt).shape)                 
    print('step4')
    load_data=True

data_mid=10

for i in range(12):
    images=np.load('Plant_Seedlings_Classification\\plant_new_data_3%s.npy'%(i))
#     print(images.shape,np.max(images),np.min(images),'done!')
    y_train_0=np.ones((len(images),1))*i
    if i ==0:
        x_train=images[:-data_mid]
        x_test=images[-data_mid:]
        y_train_=y_train_0[:-data_mid]
        y_test_=y_train_0[-data_mid:]
    else:
        x_train=np.vstack((x_train,images[:-data_mid]))
        x_test=np.vstack((x_test,images[-data_mid:]))
        y_train_=np.vstack((y_train_,y_train_0[:-data_mid]))
        y_test_=np.vstack((y_test_,y_train_0[-data_mid:]))
print(y_train_.shape)
print(y_test_.shape)
#     print(x_train.shape,y_train_.shape)
y_train=np.zeros((len(y_train_),12))
y_test=np.zeros((data_mid*12,12))
for i in range(len(y_train)):
    y_train[i,int(y_train_[i])]=1
    
for i in range(len(y_test)):    
    y_test[i,int(y_test_[i])]=1
        
print(y_train[0])
print(y_train[1000])
print(y_train[-1])
print(y_test[0])
print(y_test[data_mid*6])
print(y_test[-1])

print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)
# plt.imshow(x_test[0])
# plt.show()
# plt.imshow(x_train[0])
# plt.show()
'''
im=Image.open(train_data_path+'\\'+train_name_list_sum[1]+'\\'+'0a7e1ca41.png')
out=im.resize((2560,2560))
out.save('D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\111.jpg')
images_1=mpimg.imread('D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\111.jpg')
plt.imshow(images_1)
plt.show()
'''
print('done!')

def train_(x_train,x_test,y_train,y_test):
    x=tf.placeholder(tf.float32,[None,new_width,new_high,3])
    inputs_data=tf.reshape(x, [-1,new_width,new_high, 3])
    y=tf.placeholder(tf.float32,[None,12])
    keep_prob_=tf.placeholder(tf.float32)
    lr=tf.Variable(0.0001,dtype=tf.float32)

    filters_=32
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

    layer5=tf.layers.conv2d(layer4, filters_*16, 3, 1,'same',name='layer5')
    layer5=tf.layers.batch_normalization(layer5)
    layer5=tf.maximum(0.01*layer5,layer5)
    layer5=tf.nn.dropout(layer5, keep_prob_)
    layer5=tf.layers.max_pooling2d(layer5, 2, 2,'same')

    layer6=tf.layers.conv2d(layer5, filters_*32, 3, 1,'same',name='layer6')
    layer6=tf.layers.batch_normalization(layer6)
    layer6=tf.maximum(0.01*layer6,layer6)
    layer6=tf.nn.dropout(layer6, keep_prob_)
    layer6=tf.layers.max_pooling2d(layer6, 2, 2,'same')

    layer7=tf.layers.conv2d(layer6, filters_*8, 3, 1,'same',name='layer7')
    layer7=tf.layers.batch_normalization(layer7)
    layer7=tf.maximum(0.01*layer7,layer7)
    layer7=tf.nn.dropout(layer7, keep_prob_)
    layer7=tf.layers.max_pooling2d(layer7, 2, 2,'same')

#     layer8=tf.layers.conv2d(layer7, filters_*128, 3, 1,'same',name='layer8')
#     layer8=tf.layers.batch_normalization(layer8)
#     layer8=tf.maximum(0.01*layer8,layer8)
#     layer8=tf.nn.dropout(layer8, keep_prob_)
#     layer8=tf.layers.max_pooling2d(layer8, 2, 2,'same')
   
       
    layer9=tf.reshape(layer6, [-1,2*2*filters_*32])
    
    layer10=tf.layers.dense(layer9, 1024)
    layer10=tf.layers.batch_normalization(layer10)
    layer10=tf.maximum(0.01*layer10,layer10)
    layer10=tf.nn.dropout(layer10, keep_prob_)
    
    layer11=tf.layers.dense(layer10, 1024)
    layer11=tf.layers.batch_normalization(layer11)
    layer11=tf.maximum(0.01*layer11,layer11)
    layer11=tf.nn.dropout(layer11, keep_prob_)
    
    output=tf.layers.dense(layer11,12,tf.nn.softmax)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    prediction=tf.argmax(output,1)
    prediction1=tf.argmax(y,1) 
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(output,1))
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    acc_list=[]
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(51):
            if i % 10 ==0:
                sess.run(tf.assign(lr,(10**(-i/10))*0.0001))
            for j in range(n_batch):
                feed_data_x=np.reshape(x_train[j],[1,new_width,new_high,3])
                feed_data_y=np.reshape(y_train[j],[1,12])
                for m in range(1,64):
                    feed_data_x=np.vstack((feed_data_x,np.reshape(x_train[m*72+j],[1,new_width,new_high,3])))
                    feed_data_y=np.vstack((feed_data_y,np.reshape(y_train[m*72+j],[1,12])))
#                 print(feed_data_x.shape,feed_data_y.shape)
#                 sess.run(train_step,{x:x_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size],keep_prob_:0.5})
                sess.run(train_step,{x:feed_data_x,y:feed_data_y,keep_prob_:0.5})
#                 a,b,c,d=sess.run([layer1,layer2,layer3,layer3_],{x:x_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size],keep_prob_:0.5})
#                 print(a.shape,b.shape,c.shape,d.shape)
            acc1=sess.run(acc,{x:x_test,y:y_test,keep_prob_:1.0})
            acc_list.append(acc1)
            print('第%s次训练准确率为%s'%(i,acc1))

#         prediction_out=sess.run(prediction,{x:x_prediction,keep_prob_:1.0})
    
    return acc_list

'''
name_list=['train.csv','test.csv','sample_submission.csv']
path_list=[]
for i in range(3):
    path='D:\\Desktop\\kaggle\\digits_20180122\\'+name_list[i]
    path_list.append(path)
print(path_list)
path_train=path_list[0]
path_test=path_list[1]
'''
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
    acc_list=train_(x_train,x_test,y_train,y_test)
    plt.plot(acc_list)
    plt.show()
'''   
with open('D:\\Desktop\\kaggle\\digits_20180122\\kaggle_test_201802021707.csv','w',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['Imageld','Label'])
    for i in range(len(prediction_output)):
        data=(i+1,prediction_output[i])
        writer.writerow(data)

'''
print('done!')