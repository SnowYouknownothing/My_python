# -*- coding: utf-8 -*-
'''
Created on 2017年12月28日
@author: Administrator
'''
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

# x_plt=np.transpose(np.reshape(x_train,[50000,3,32,32]),[0,2,3,1])
# x_3=np.reshape(x_train,[50000,3,32,32])[0,0]
# print(x_3.shape)
# plt.imshow(x_3)
# # plt.show()
# 
# x_33=np.reshape(x_train,[50000,3,32,32])[0,1]
# print(x_33.shape)
# plt.imshow(x_33)
# # plt.show()
# x_333=np.reshape(x_train,[50000,3,32,32])[0,2]
# print(x_333.shape)
# plt.imshow(x_333)
# plt.show()
# print(x_plt.shape)
# plt.imshow(x_plt[0])
# plt.show()


print(max(x_train[0]),min(x_train[0]))




# x_1=np.transpose(np.reshape(x_train,[50000,32,32,3]),[0,1,2,3])
# 
# # x_2=np.reshape(x_train,[50000,32,3,32])
# 
# print(x_1.shape)
# plt.imshow(x_1[0])
# plt.show()
