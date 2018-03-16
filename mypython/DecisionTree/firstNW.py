# -*- coding: utf-8 -*-
'''
Created on 2017年11月21日
@author: Administrator
'''
import numpy as np
from numpy import exp, array, random, dot
# from numpy import random,array
def sigmord(x):
    return 1/(1+exp(-x))
def error(a,b):
    return a*(1-a)*(b-a)    

learningrate=0.2
x=[]
y=[]
# np.random.seed(1)
for i in range(1000):
    a=random.random()
    b=random.random()
    c=random.random()
    d=(a+b+c)/3
    x.append([a,b,c])
    y.append([d])
x=array(x)
y=array(y)
w=random.random((len(x[1]),1))
b=random.random((len(y),1))
# print x
# print y
# print w
# print b
#     l_j=0
for xuanhuancishu in range(10000):
    l_j = dot(x,w)+b
    o_j = sigmord(l_j)
#     算出output
    error_j=error(o_j,y)
    w+=learningrate*dot(x.T,error_j)
    b+=learningrate*error_j
# 
# print error_j
# print '\n'
# print w
# print b
# print b
yuce_input=[random.random(),random.random(),random.random()]
yuce_output=sigmord(dot(yuce_input,w))
zhunquelv=sum(yuce_output)/(sum(yuce_input)/3)
print yuce_output
print zhunquelv
    
        
#     print l_j,o_j,error_j,w[1],b
    
    
    
    
    

    
    
    
    
    
    
    