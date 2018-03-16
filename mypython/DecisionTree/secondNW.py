# -*- coding: utf-8 -*-
'''
Created on 2017年11月21日
@author: Administrator
'''
import numpy as np
from numpy import exp, array, random, dot
# from numpy import random,array
def sigmord(x):
    return 1/(1+np.exp(-x))
def error(x,y):
    return x*(1-x)*(y-x)    

learningrate=0.1
weizhi=0
x=np.array([[1,0,1]])
y=array([1])
np.random.seed(1)
w=array([np.random.random(1),np.random.random(1),np.random.random(1)])
b=array([np.random.random(1)])
# b=array([np.random.random(1),np.random.random(1),np.random.random(1),np.random.random(1)])
print x
print y
print w
print dot(x,w)
# print b
#     l_j=0
for xuanhuancishu in range(10000):
    l_j = dot(x,w)+b
    o_j = sigmord(l_j)
#     算出output
    error_j=error(o_j,y)
    w+=learningrate*error_j*o_j
    b+=learningrate*error_j
print error_j,b
print '\n'
print w
# print b
yuce_input=[0,1,0]
yuce_output=dot(yuce_input,w)
print yuce_output