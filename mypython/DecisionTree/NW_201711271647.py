# -*- coding: utf-8 -*-
'''
Created on 2017年11月27日
@author: Administrator
'''
from numpy import random,exp,dot,eye,array
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
def sigmord(x):
    return 1/(1+exp(-x))
def tanh(x):
    return np.tanh(x)
def sigmord_deriv(x):
    return sigmord(x)*(1-sigmord(x))
def tanh_deriv(x):
    return 1.0-np.tanh(x)*np.tanh(x)
def error_output(o,t):
    return o*(1-o)*(t-o)
def train(learning_rate,epochs,daying,x_input,y_output,layel):          
    w,b=[],[]
    for i in range(len(layel)-1):
        w.append(2*random.random((layel[i],layel[i+1])))
        b.append(2*random.random((layel[i+1],1)))
    w=array(w)-1
    b=array(b)-1     
    less,i_x=[],[] 
    for j in range(epochs):
        number_random=random.randint(len(x_input))
        x_input_suiji=eye(1,len(x_input[0]))
        for i in range(len(x_input[0])):
                x_input_suiji[0,i]=x_input[number_random,i]
        y_output_suiji=y_output[number_random]
#随机训练数
        l_x,sigmord_x=[],[]
        sigmord_x.append(x_input_suiji)
        for i in range(len(layel)-1):
            l_x.append(dot(sigmord_x[i],w[i])+ b[i].T)
            sigmord_x.append(sigmord(l_x[i]))
#前向传播结束      
        error_x=[]
        error_x.append((sigmord_deriv(l_x[-1]))*(y_output_suiji-sigmord_x[-1]))
        for i in range(len(layel)-2):
            error_x.append((sigmord_deriv(l_x[-(i+2)]))*dot(error_x[i],w[-(i+1)].T))  
# 反向传播结束
        for i in range(len(layel)-1):
            w[i]+=dot(sigmord_x[i].T,(learning_rate*error_x[-(i+1)]))
            b[i]+=learning_rate*(error_x[-(i+1)].T)
# 更新w,b
        i_x.append(j)
        less.append(float(sigmord_x[-1]-y_output_suiji))
# 统计less
    if daying=='T':
        plt.figure(figsize=(100,100))
        plt.plot(i_x,less,label="$less$",color="red",linewidth=0.5)
        plt.ylim(-1,1)
        plt.xlabel("epochs")
        plt.ylabel("less")
        plt.legend()
        plt.show()
# 画less图
    return w,b
def predict():
    return 0
digits=load_digits()
x=digits.data
y=digits.target
number_index=1000
x_input,y_output=x[:number_index,:len(x[0])]*0.1,y[:number_index]*0.1
x_kaohe,y_kaohe=x[number_index+1:len(x),:len(x[0])]*0.1,y[number_index+1:len(y)]*0.1
for i in range(1):
    epochs=500000
    learning_rate=9
    daying='T'
    layel=[len(x_input[0]),30,30,30,1]  
    w,b=train(learning_rate,epochs,daying,x_input,y_output,layel)
    sum_right,sum_x=[],[]
    sum_1,sum_2=0,0
    for j in range(len(y_kaohe)):
        x_input_random=eye(1,len(x_input[0]))
        for i in range(0,len(x_input[0])):
            x_input_random[0,i]=x_kaohe[j,i]
        l_x,sigmord_x=[],[]
        sigmord_x.append(x_input_random)
        for i in range(len(layel)-1):
            l_x.append(dot(sigmord_x[i],w[i])+ b[i].T)
            sigmord_x.append(sigmord(l_x[i]))
        y_difference=sigmord_x[-1]-y_kaohe[j]    
        sum_x.append(j)
        if y_kaohe[j]!=0.0:
#             print '第%s次预测数：%s，真实数为：%s，准确率为：%s'%(j+1,float(sigmord_x[-1]),y_kaohe[j],float(y_difference/y_kaohe[j]))
            sum_right.append(float(y_difference/y_kaohe[j]))
        else:
#             print '第%s次预测数：%s，真实数为：%s'%(j+1,float(y_o),y_ture)
            sum_right.append(y_kaohe[j])
        if y_difference>-0.05 and y_difference<0.05:
            sum_1+=1
        if y_difference>-0.1 and y_difference<0.1:
            sum_2+=1

    print '准确率百分之95以上的有%s个' %(sum_1)
    print '准确率百分之90以上的%s个' %(sum_2)
    plt.figure(figsize=(100,100))
    plt.plot(sum_x,sum_right,label="$sum_right$",color="red",linewidth=1)
    plt.ylim(-1,1)
    plt.xlabel("i_x1")
    plt.ylabel("sum_right")
    plt.legend()
    plt.show()
      
    plt.title("scatter diagram")
    plt.xlim(xmax=800,xmin=0)
    plt.ylim(ymax=1.5,ymin=-1)
    plt.xlabel("sum_x")
    plt.ylabel("zhunquelv")
    plt.scatter(sum_x,sum_right,s=5,alpha=0.9,marker='o')
    plt.show()   
print'测试结束'







