# -*- coding: utf-8 -*-
'''
Created on 2017年12月15日
@author: Administrator
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minist=input_data.read_data_sets('Minist_data',one_hot=True)
x=minist.test.images
y=minist.train.images
print(x.shape,y.shape)