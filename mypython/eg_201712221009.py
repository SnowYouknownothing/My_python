# -*- coding: utf-8 -*-
'''
Created on 2017年12月22日
@author: Administrator
'''
from numpy import array, reshape
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
# from matplotlib.pyplot import autumn

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
x_train,y_train=[],[]
for i in range(1,6):
    datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//data_batch_%s'%(i))
    for j in range(len(datadict[b'data'])):
        x_train.append(datadict[b'data'][j])
        y_train.append(datadict[b'labels'][j])
# print(len(x_test),len(x_test[0]),len(y_test))
 
datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//test_batch')
x_test,y_test=datadict[b'data'],datadict[b'labels']
print(len(x_test),len(x_test[0]),len(y_test))
print(len(x_train),len(y_train))
print(max(x_train[0]),min(x_train[0]))
x_train=np.reshape(x_train, [50000,3,32,32] )
x_train=np.transpose(x_train, [0,2,3,1])

print(x_train[0,0])

fig,axes=plt.subplots(nrows=5, ncols=20, sharex=True, sharey=True, figsize=(20,5))
imgs=x_train[300:400]
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()
# plt.imshow(x_train[4])
# plt.show()





# from tensorflow.examples.tutorials.mnist import input_data
# minst=input_data.read_data_sets('minst_data')
# x=minst.train.images
# x=np.reshape(x,[x.shape[0],28,28])







