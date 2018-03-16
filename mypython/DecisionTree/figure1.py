# -*- coding: utf-8 -*-
'''
Created on 2017年11月23日
@author: Administrator
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy import random,exp,dot,eye,shape, reshape
from sklearn.datasets import load_digits

# x = np.linspace(0, 100, 10000)
# y = np.sin(x)
# z = np.cos(x**2)
# 
# plt.figure(figsize=(8,4))
# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=1)
# plt.ylim(-1.2,1.2)
# plt.legend()
# plt.show()
digits=load_digits()
# x=digits.data
# y=digits.target
# x_input=x[:1000,:64]*0.1
# y_output=y[:1000]*0.1
# x_kaohe=x[1001:len(x),:64]*0.1
# y_kaohe=y[1001:len(y)]*0.1
# print len(x_input)
# print len(y_output)
# print len(x_kaohe)
# print len(y_kaohe)
# print len(x_input[0])
print digits.data.shape
print digits.images[5]
print digits.data[5]
print digits.target[111]
import pylab as pl
pl.gray()
pl.matshow(digits.images[5])
pl.show()


x=reshape(digits.data[5],[8,8])
pl.gray()
pl.matshow(x)
pl.imshow(x)
pl.show()








