# -*- coding: utf-8 -*-
'''
Created on 2017年11月22日
@author: Administrator
'''
from numpy import array,dot,random,exp
def sigmord(x):
    return 1/(1+exp(-x))
def error_output(o,t):
    return o*(1-o)*(t-o)
x_input=[]
y_target=[]
learning_rate=0.2
random.seed(1)
x_input=random.random((4,3))
for i in range(x_input.shape[0]):
    y_target.append(sum(x_input[i])/3)
y_target=array(y_target)
w_ij=2*random.random((2,3))-1
w_jo1=2*random.random()-1
w_jo2=2*random.random()-1
b=2*random.random((3,1))-1
def FP(x,w1,w2,w3,b):
    lj1=dot(x,w1[0])+b[0]
    sigmord_lj1=sigmord(lj1)
    lj2=dot(x,w1[1])+b[1]
    sigmord_lj2=sigmord(lj2)
    lj3=(sigmord_lj1 *w2) + (sigmord_lj2 *w3) + b[2]
    output=sigmord(lj3)
    return output,sigmord_lj1,sigmord_lj2
y_output,sigmord_lj1,sigmord_lj2=FP(x_input,w_ij,w_jo1,w_jo2,b)
# print y_output,sigmord_lj1,sigmord_lj2
# y_output=array(y_output)
# y_output=y_output.T
error_output1=error_output(y_output,y_target)
w_jo1+=learning_rate*error_output1*sigmord_lj1
w_jo2+=learning_rate*error_output1*sigmord_lj2

    
    
    
    
    
    
    
    
