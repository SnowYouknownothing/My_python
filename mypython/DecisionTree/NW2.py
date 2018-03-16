# -*- coding: utf-8 -*-
'''
Created on 2017年11月22日
@author: Administrator
'''
from numpy import random,exp
def sigmord(x):
    return 1/(1+exp(-x))
def error_output(o,t):
    return o*(1-o)*(t-o)

number_shujuliang=100
epochs=100000
learning_rate=0.1

for i in range(20):

    x_input=random.random((number_shujuliang,3))
    for i in range(number_shujuliang):
        for j in range(3):
            x_input[i,j]=round(x_input[i,j],2)     
    
    y_target=[]
    for i in range(number_shujuliang):
        y_target.append((sum(x_input[i]))/3)
    # x_input=array([[0.2,0.3,0.5],[0.4,0.2,0.6],[0.3,0.3,0.3],[0.7,0.2,0.3]])
    # y_target=array([[1/3],[0.4],[0.3],[0.4]])
    
    # random.seed(1)
    w_i1j1=round((2*random.random()-1),2)
    w_i1j2=round((2*random.random()-1),2)
    w_i2j1=round((2*random.random()-1),2)
    w_i2j2=round((2*random.random()-1),2)
    w_i3j1=round((2*random.random()-1),2)
    w_i3j2=round((2*random.random()-1),2)
    w_j1o=round((2*random.random()-1),2)
    w_j2o=round((2*random.random()-1),2)
    b_j1=0
    b_j2=0
    b_o=0
    
    for i in range(epochs):
        number=random.randint(number_shujuliang)
        x_1,x_2,x_3=x_input[number]
        # 前向传播
        l_j1_1=x_1*w_i1j1+x_2*w_i2j1+x_3*w_i3j1+b_j1
        
        l_j2_1=x_1*w_i1j2+x_2*w_i2j2+x_3*w_i3j2+b_j2
        
        sigmord_l_j1_1=sigmord(l_j1_1)
        
        sigmord_l_j2_1=sigmord(l_j2_1)
        
        o_1=sigmord_l_j1_1*w_j1o+sigmord_l_j2_1*w_j2o+b_o
        # 完成前向传播
        error_o_1=error_output(o_1,y_target[number])
        error_j1=sigmord_l_j1_1*(1-sigmord_l_j1_1)*error_o_1*w_j1o
        error_j2=sigmord_l_j2_1*(1-sigmord_l_j2_1)*error_o_1*w_j2o
        # o层更新完毕
        
        w_i1j1+=learning_rate * x_1 * error_j1
        w_i1j2+=learning_rate * x_1 * error_j2
        w_i2j1+=learning_rate * x_2 * error_j1
        w_i2j2+=learning_rate * x_2 * error_j2
        w_i3j1+=learning_rate * x_3 * error_j1
        w_i3j2+=learning_rate * x_3 * error_j2
        
        b_j1+=learning_rate *error_j1
        b_j2+=learning_rate *error_j2
        
        w_j1o+=learning_rate*error_o_1*sigmord_l_j1_1
        w_j2o+=learning_rate*error_o_1*sigmord_l_j2_1
        b_o+=learning_rate*error_o_1
    
    # print w_i1j1
    # print w_i1j2 
    # print w_i2j1
    # print w_i2j2
    # print w_i3j1
    # print w_i3j2
    # print b_j1
    # print b_j2
    # print b_o
    
    x_1,x_2,x_3=[round((random.random()),2),round((random.random()),2),round((random.random()),2)]
        # 前向传播
    l_j1_1=x_1*w_i1j1+x_2*w_i2j1+x_3*w_i3j1+b_j1
        
    l_j2_1=x_1*w_i1j2+x_2*w_i2j2+x_3*w_i3j2+b_j2
        
    sigmord_l_j1_1=sigmord(l_j1_1)
        
    sigmord_l_j2_1=sigmord(l_j2_1)
        
    o_1=sigmord_l_j1_1*w_j1o+sigmord_l_j2_1*w_j2o+b_o
    
#     print o_1
    print o_1/((x_1+x_2+x_3)/3)








