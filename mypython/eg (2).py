# -*- coding: utf-8 -*-
'''
Created on 2017年11月30日
@author: Administrator
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
minst=input_data.read_data_sets('Minst_data')
bacth_size=100
n_bacth=minst.train.num_examples//bacth_size
# print(n_bacth,minst.train.images.shape,minst.train.labels.shape)
x,y=minst.test.images,minst.test.labels
# print(len(x),len(y),x.shape,y.shape)
# print(x[0],y[0])
# 
# print(minst.test.images.shape)
# import pylab as pl    
# pl.gray()
# pl.matshow(minst.train.images)
# pl.show()
# for i in range(1):
#     x_data,y_data=minst.train.next_batch(1)
#     print(x_data,y_data,x_data.shape,y_data.shape)
#     import pylab as pl    
#     pl.gray()
#     pl.matshow(x_data)
#     pl.show()


