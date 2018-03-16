# -*- coding: utf-8 -*-
'''
Created on 2017年12月7日
@author: Administrator
'''
import tensorflow as tf  
#   
# a = tf.constant(5, name="input_a")  
# b = tf.constant(3, name="input_b")  
# c = tf.multiply(a, b, name="mul_c")  
# d = tf.add(a, b, name="add_d")  
# e = tf.add(c, d, name="add_e")  
#   
# sess = tf.Session()  
# sess.run(e)  
#   
# writer = tf.summary.FileWriter("F:/tensorflow/graph", tf.get_default_graph())  
# writer.close() 
# print('done')
# a={1:2,2:3}
# print(a[0])

# from numpy import array
# import tensorflow as tf
# import numpy as np
# from sklearn.datasets import load_digits
# digits=load_digits()
# x=digits.data
# y=digits.target
# print(x.shape,y.shape,max(x[0]),min(x[0]),y[0],x[0])

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         data = pickle.load(fo, encoding='bytes')
#     return data
# x_test,y_test=[],[]
# for i in range(1,6):
#     datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//data_batch_%s'%(i))
#     for j in range(len(datadict[b'data'])):
#         x_test.append(datadict[b'data'][j])
#         y_test.append(datadict[b'labels'][j])
# print(len(x_test),len(x_test[0]),len(y_test))
# datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//test_batch')
# x_train,y_train=datadict[b'data'],datadict[b'labels']
# print(len(x_train),len(y_train))
# 
# x_test,y_test1,x_train,y_train1=array(x_test),array(y_test),array(x_train),array(y_train)
# print(max(x_test[0]),min(x_test[0]))
# print(x_test.shape,y_test1.shape,x_train.shape,y_train1.shape)
# 
# y_train=np.eye(10000,10)*0
# for i in range(10000):
#     y_train[i,y_train1[i]]=1          
# y_test=np.eye(50000,10)*0
# for i in range(50000):
#     y_test[i,y_test1[i]]=1 
# 
# print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)
# print(x_test[0],y_test[0])
# print(x_train[0],y_train[0])

from numpy import array
# import tensorflow as tf
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# 
# 
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         data = pickle.load(fo, encoding='bytes')
#     return data
# x_test,y_test=[],[]
# for i in range(1,6):
#     datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//data_batch_%s'%(i))
#     for j in range(len(datadict[b'data'])):
#         x_test.append(datadict[b'data'][j])
#         y_test.append(datadict[b'labels'][j])
# print(len(x_test),len(x_test[0]),len(y_test))
#  
# datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//test_batch')
# x_train,y_train=datadict[b'data'],datadict[b'labels']
# print(len(x_train),len(y_train))
# x_test,y_test1,x_train,y_train1=array(x_test),array(y_test),array(x_train),array(y_train)
#  
# eg1=x_test[6000]
# eg1=eg1.reshape(32,32,3)
# plt.imshow(eg1)
# plt.show()


import numpy as np

# a=np.random.uniform(-1,2,3)[:,np.newaxis]
# 
# b=np.random.random((3,3))
# print(a.shape,b.shape)
# 
# c=a*b 
# print(a)
# print('\n')
# print(b)
# print('\n')
# print(c)
# print('\n')
# print(a[0,0]*b[0])
# print(c)




from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
minst=input_data.read_data_sets('minst_data')
x_real=minst.test.images
y_real=minst.test.labels
print(x_real.shape,y_real.shape)
 
print(max(x_real[10]),min(x_real[10]))

real=np.reshape(x_real[10],[28,28])

print(x_real[13])
plt.figure()
plt.imshow(real)
# plt.imshow(real*255)
plt.show()

