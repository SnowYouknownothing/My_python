# -*- coding: utf-8 -*-
'''
Created on 2017年12月28日
@author: Administrator
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_inputs=1/(1+np.exp(-np.linspace(-10, 10, 100)))
plt.plot(x_inputs)
plt.show()


x_real=tf.placeholder(tf.float32,[None,10])
x_fake=tf.placeholder(tf.float32,[None,10])

# with tf.variable_scope('G'):
    
