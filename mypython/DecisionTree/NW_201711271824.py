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
class NeuralNetwork:
    def __init__(self,layel):
        self.w,self.b=[],[]
        for i in range(len(layel)-1):
            self.w.append(2*random.random((layel[i],layel[i+1])))
            self.b.append(2*random.random((layel[i+1],1)))
        self.w=array(self.w)-1
        self.b=array(self.b)*0     
    def train(self,learning_rate,epochs,x_train,y_train,layel):               
        less,i_x=[],[] 
        for j in range(epochs):
#             learning_rate=9+(learning_rate/epochs)*j
            number_random=random.randint(len(x_train))
            x_input_random=eye(1,len(x_train[0]))
            for i in range(len(x_train[0])):
                    x_input_random[0,i]=x_train[number_random,i]
            y_output_random=y_train[number_random]
    #随机训练数
            l_x,sigmord_x=[],[]
            sigmord_x.append(x_input_random)
            for i in range(len(layel)-1):
                l_x.append(dot(sigmord_x[i],self.w[i])+ self.b[i].T)
                sigmord_x.append(sigmord(l_x[i]))
    #前向传播结束      
            error_x=[]
            error_x.append((sigmord_deriv(l_x[-1]))*(y_output_random-sigmord_x[-1]))
            for i in range(len(layel)-2):
                error_x.append((sigmord_deriv(l_x[-(i+2)]))*dot(error_x[i],self.w[-(i+1)].T))  
    # 反向传播结束
            for i in range(len(layel)-1):
                self.w[i]+=dot(sigmord_x[i].T,(learning_rate*error_x[-(i+1)]))
                self.b[i]+=learning_rate*(error_x[-(i+1)].T)
    # 更新w,b
            i_x.append(j)
            less.append(float(sigmord_x[-1]-y_output_random))
#             if j ==1/4*epochs:
#                 if less[j-1]>0.25 or less[j-1]<-0.25:
#                     break
    # 统计less
        plt.figure(figsize=(100,100))
        plt.plot(i_x,less,label="$less$",color="red",linewidth=0.5)
        plt.ylim(-1,1)
        plt.xlabel("epochs")
        plt.ylabel("less")
        plt.legend()
        plt.show()
    # 画less图
    def predict(self,x_test):
        y_test=[]
        for j in range(len(x_test)):
            x_input_random=eye(1,len(x_test[0]))
            for i in range(0,len(x_test[0])):
                x_input_random[0,i]=x_test[j,i]
            l_x,sigmord_x=[],[]
            sigmord_x.append(x_input_random)
            for i in range(len(layel)-1):
                l_x.append(dot(sigmord_x[i],self.w[i])+ self.b[i].T)
                sigmord_x.append(sigmord(l_x[i]))
            y_test.append(sigmord_x[-1])
        y_test=array(y_test)       
        return y_test

digits=load_digits()
x=digits.data
y=digits.target
train_index=1000
x_train,y_train=x[:train_index,:len(x[0])]/16,y[:train_index]*0.1
x_test,y_target=x[train_index+1:len(x),:len(x[0])]/16,y[train_index+1:len(y)]*0.1



epochs=10000
learning_rate=4
layel=[len(x_train[0]),60,50,40,1]  

nw=NeuralNetwork(layel)
nw.train(learning_rate, epochs, x_train, y_train, layel)
y_test=nw.predict(x_test)

y_difference,x_difference=[],[]
sum_right=0
for i in range(len(y_target)-1):
    y_difference.append(y_test[i,0,0]-y_target[i])
    x_difference.append(i)
    if float(y_test[i,0,0]-y_target[i])>-0.01 and float(y_test[i,0,0]-y_target[i])<0.01:
        sum_right+=1
print sum_right        
print float(sum_right/len(y_target))

# plt.figure(figsize=(100,100))
# plt.plot(x_difference,y_difference,label="$y_difference$",color="red",linewidth=1)
# plt.ylim(-1,1)
# plt.xlabel("x")
# plt.ylabel("y_difference")
# plt.legend()
# plt.show()
#   
plt.title("scatter diagram")
plt.xlim(xmax=800,xmin=0)
plt.ylim(ymax=1.5,ymin=-1)
plt.xlabel("x")
plt.ylabel("y_difference")
plt.scatter(x_difference,y_difference,s=5,alpha=0.9,marker='o')
plt.show()   
print'测试结束'







