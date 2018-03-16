# -*- coding: utf-8 -*-
'''
Created on 2018年1月15日
@author: Administrator
'''
import numpy as np

def gradientDescent(x,y,theta=np.ones(2),alpha=0.0005,m=100,numIterations=100000):
    xTrans=x.T
    for i in range(numIterations):
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        cost=np.sum(loss**2)/(2*m)
        if i%(numIterations/10)==0:
            print('第%s次，cost:%s'%(i,cost))
        gradient=(np.dot(xTrans,loss))/m
        theta=theta-alpha*gradient
    return theta
        
        
def getDate(num=100):
    x,y=np.zeros([num,2] ),np.zeros(num)
    for i in range(num):
        x[i,0]=1
        x[i,1]=i
        y[i]=i+25+np.random.uniform(0,1)*5
    return x,y
x,y=getDate()
# theta=np.ones(100)
print(x,y)
theta=gradientDescent(x, y)
print(theta)

