# -*- coding: utf-8 -*-
'''
Created on 2018年1月24日
@author: Administrator
'''

from numpy import array, reshape
import tensorflow as tf
import numpy as np
import time
import csv 
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image

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

x_train=np.array(x_train)
y_train=np.array(y_train)

print(x_train.shape,y_train.shape)

x_images=np.transpose(np.reshape(x_train,[len(x_train),3,32,32]),[0,2,3,1])

print(x_images.shape)

for i in range(5000):   
    scipy.misc.imsave('D:\\Desktop\\kaggle\\cifar\\%s.jpg'%((int(y_train[i]))*10000+i,x_images[i]))

print('done!')

