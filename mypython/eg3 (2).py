# -*- coding: utf-8 -*-
'''
Created on 2017年11月16日
@author: Administrator
'''
import numpy as np
from numpy import exp, array, random, dot,eye,shape
import sklearn
from sklearn.datasets import load_digits
# from matplotlib.pyplot import plot
# random.seed(1)
# # w=np.random.randn(5,10)
# # print w
# # print '\n'
# # x=np.random.randn(10,3)
# # print x
# # print '\n'
# # d=w.dot(x)
# # print d
# # print '\n'
# # e=dot(w,x)
# # print e
# # print '\n'
# # for i in range(1,5):
# #     print i
# f=[]
# f.append((2*(np.random.random((11,11)))-1)*0.25)
# print f
# a=array([1,2,3])
# b=([2,3,4])
# print a*b
# training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# print len(training_set_inputs[0])
# print training_set_inputs
# print training_set_inputs.T
# y=array([9,11,21,9])
# print y*(1-y)
# for i in range(1,2):
#     print i
# b=random.random(3)
# a=random.random((5,4))
# print a
# print b
# x=array([[1,2,3],[2,3,4],[3,4,5],[3,4,5]])
# print np.random.randint(x.shape[0]),len(x)
# print x ,x.shape[0]
# temp=np.ones([x.shape[0],x.shape[1]+1])
# print temp
# print temp[:,0:3 ]
# y=[[1],[2],[3]]
# print y
# print array(y)
# print x[1]
# print x[0,0:3]


# x=np.array([[3,3,3],[4,3,4],[7,8,6],[4,3,2],[2,3,2],[7,5,8],[3,7,6]])
# y=array([[9],[11],[21],[9],[7],[20],[16]])
# x=[1,2,3]
# print sum(x)
# y=random.random()
# print y
# np.random.seed(1)
# for i in range(50):
#     a=random.randint(100)
#     b=random.randint(100)
#     c=random.randint(100)
#     d=a+b+c
#     x.append([a,b,c])
#     y.append([d])
# print x
# print y
# x=array(x)
# y=array(y)
# print x
# print y

# b=2*random.random((3,1))-1
# print b
# print b[0]
# print b[1]
# a=b+2
# print a
# c=2*random.random((3,1))-1
# d=b+c
# print d
# x=array([[1,2,3],[2,3,4],[3,4,5],[3,4,5]])
# a,b,c=x[2]
# print a,b,c
# 
# for i in range(100):
#     number=random.randint(4)
#     print number
# x_input=array([[0.2,0.3,0.5],[0.4,0.2,0.6],[0.3,0.3,0.3],[0.7,0.2,0.3]])
# y_target=array([[1/3],[0.4],[0.3],[0.4]])
# y=x_input[1:3,1:3]
# print x_input[2]
# print x_input[0,0]
# for i in range(100):
#     number=random.randint(4)
#     x_1,x_2,x_3=x_input[number]
#     print x_1,x_2,x_3
#     print random.randint(2)
# number_shujuliang=10
# epochs=1000000
# learning_rate=0.1
# 
# x_input=random.random((number_shujuliang,3))
# y_target=[]
# print x_input
# for i in range(number_shujuliang):
#     print i
# #     y_target.append((sum(x_input[i]))/3)
#      y_target.append([x_input[i-1,0]])
     
# a=random.random(8)   
# print a
# c=random.randint(10)
# print c

# import time
# plt.ion() #开启interactive mode
# x = np.linspace(0, 50, 1000)
# plt.figure(1) # 创建图表1
# plt.plot(x, np.sin(x))
# plt.draw()
# time.sleep(5)
# plt.close(1)
# plt.figure(2) # 创建图表2
# plt.plot(x, np.cos(x))
# plt.draw()
# time.sleep(50)
# print 'it is ok'
# from pylab import import *
# x_input=array([[0.2,0.3,0.5]])
# print sum(x_input)

# from sklearn.datasets import load_digits
# digits=load_digits()
# x=digits.data
# y=digits.target
# # print x
# # print len(x)
# # print y
# # print len(y)
# x_ceshi=x[:1000,:64]
# y_ceshi=y[:1000]*0.1
# print y_ceshi
# 
# print len(x_ceshi)
# print len(y_ceshi)


# import numpy as np
# import matplotlib.pyplot as plt
# 
# 
# 
# plt.figure(figsize=(10,10))
# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=1)
# plt.ylim(0,25)
# plt.legend()
# plt.show()
# y=random.random((2,2))
# print y.shape,type(y)
# x=[]
# x.append(np.random.random((5,4)))
# print x
# print'\n'
# x.append(np.random.random((4,3)))
# print x
# print'\n'
# x.append(np.random.random((3,2)))
# print x
# print'\n'
# x.append(np.random.random((2,1)))
# x=array(x)
# print x
# print'\n'
# print x.shape
# print type(x),len(x),len(x[0]),len(x[1])
# print x[0]
# i=eye(5,1)
# print i
# print i[2]
# for i in range(5,0,-1):
#     print i
# b=[]
# b.append(2*random.random((3,2))) 
# print b
# a=[1,2,3,4]
# print a[-1],a[-2],a[-3]
# digits=load_digits()
# x=digits.data
# y=digits.target
# # print len(x[0])
# f=open('first.txt','r')
digits=load_digits()
x=digits.data
y=digits.target
 
print x.shape,y.shape
print max(x[0]),max(y)
print min(x[0]),min(y)
print '\n'
for i in range(64):
    print max(x[0:len(x),i]),min(x[0:len(x),i])
# x=random.random((20,40000))
# print max(x[0]),len(x[0])


print'\n'
print np.random.rand(20)
print np.random.random(20)

