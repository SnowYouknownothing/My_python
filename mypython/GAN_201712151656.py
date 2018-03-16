# -*- coding: utf-8 -*-
'''
Created on 2017年12月7日
@author: Administrator
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from numpy import reshape
minst=input_data.read_data_sets('Minst_data',one_hot=True)
x=minst.test.images
print(x.shape)

y=reshape(x,[10000,28,28])

import pylab as pl
pl.gray()
pl.matshow(y[11])
pl.show()