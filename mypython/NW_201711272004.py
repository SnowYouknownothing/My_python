# -*- coding: utf-8 -*-
'''
Created on 2017年11月27日
@author: Administrator
'''
import numpy as np
import cPickle as pickle
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y
filename='D://Desktop//cifar-10-batches-py//'
x,y=load_CIFAR_batch(filename)
print x,y