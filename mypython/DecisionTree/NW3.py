# -*- coding: utf-8 -*-
'''
Created on 2017年11月22日
@author: Administrator
'''
from numpy import random,exp,dot,eye
import matplotlib.pyplot as plt

def sigmord(x):
    return 1/(1+exp(-x))

def error_output(o,t):
    return o*(1-o)*(t-o)

def PD(learning_rate,epochs,x_yuce,daying,j_index,x_input,y_output):
#     index_hang=hang
#     index_lie=lie
    index_hang=1000
    index_lie=64
#     epochs=epochs_input
#     learning_rate=learning_rate_input
    layel=[index_lie,j_index,1]
#     x_input=random.random((index_hang,index_lie))
#     y_output=[]
#     for i in range(index_hang):
#         y_output.append(sum(x_input[i])/index_lie)
    w_ij=2*random.random((layel[0],layel[1]))-1
    w_jo=2*random.random((layel[1],layel[2]))-1
    b_j=2*random.random((layel[1],1))-1
    b_o=2*random.random((layel[2],1))-1
    
    less=[]
    i_x=[]
    
    for j in range(epochs):
        number_suiji=random.randint(index_hang)
        x_input_suiji=eye(1,index_lie)
        # print x_input_suiji,x_input_suiji.shape
#         if j==epochs-1:
#             for i in range(0,index_lie):
#                 x_input_suiji[0,i]=x_yuce[i]
#            
#         else:
#             for i in range(0,index_lie):
#                 x_input_suiji[0,i]=x_input[number_suiji,i]
        for i in range(0,index_lie):
                x_input_suiji[0,i]=x_input[number_suiji,i]
            
#         x_input_suiji,(1,3)
        y_output_suiji=y_output[number_suiji]
        
#         y_output_suiji=y_output[number_suiji]
#          y_output_suiji,(1,1)   
        
        l_j=dot(x_input_suiji,w_ij)+ b_j.T
        # print w_ij
        # print l_j,l_j(1,2)
        
        sigmord_j=sigmord(l_j)
        l_o=dot(sigmord_j,w_jo)+b_o
        y_o=sigmord(l_o)
#         l_o(1,1)
        
        # 正向传播结束
        # print sigmord_j
        
        error_o=error_output(y_o,y_output_suiji) 
#         error_o(1,1)
        
        
        error_j=sigmord_j*(1-sigmord_j)*error_o*(w_jo.T)
#         sigmord_j和w_jo.T一样
        # error_j(1,2)
        # print error_j
        # print x_input_suiji
         
        w_ij+=dot(x_input_suiji.T,learning_rate*error_j)
        
        
        b_j+=learning_rate*(error_j.T)
        w_jo+=learning_rate*error_o*(sigmord_j.T)
        b_o+=learning_rate*error_o
       
        i_x.append(j)
#         if less
        less.append(float(y_o-y_output_suiji))
        # print w_ij
        # print b_j
        # print w_jo
        # print b_o
#         print less
#         print 

    for i in range(0,index_lie):
        x_input_suiji[0,i]=x_yuce[i]
    l_j=dot(x_input_suiji,w_ij)+ b_j.T       
    sigmord_j=sigmord(l_j)
    l_o=dot(sigmord_j,w_jo)+b_o
    y_o=sigmord(l_o)

    if daying=='T':
        plt.figure(figsize=(100,100))
        plt.plot(i_x,less,label="$less$",color="red",linewidth=0.5)
        plt.ylim(-1,1)
        plt.xlabel("epochs")
        plt.ylabel("less")
        plt.legend()
        plt.show()

#     print less
#     print i_x
    return y_o
# for i in range(1):
#     lie=10
#     hang=100
#     epochs_input=10000
#     learning_rate_input=1
#     daying='f'
#     j_index=10
#     x_yuce=random.random((lie,1))
#     y_ture=sum(x_yuce)/lie
#     y_yuce=PD(learning_rate_input,epochs_input,hang,lie,x_yuce,daying,j_index)
#     print '第%s次准确率：%s'%(i+1,float(y_yuce/y_ture))


from sklearn.datasets import load_digits
digits=load_digits()
x=digits.data
y=digits.target
x_ceshi=x[:1000,:64]*0.1
y_ceshi=y[:1000]*0.1
x_kaohe=x[1001:len(x),:64]*0.1
y_kaohe=y[1001:len(y)]*0.1
for i in range(20):
    lie=64
    hang=1000
    epochs=200000
    learning_rate=1
    daying='T'
    j_index=100
    x_input=x_ceshi
    y_output=y_ceshi    
    index_suiji=random.randint(len(y_kaohe)) 
    x_yuce=x_kaohe[index_suiji]
    y_ture=y_kaohe[index_suiji]
    y_yuce=PD(learning_rate,epochs,x_yuce,daying,j_index,x_input,y_output)
#     print '第%s次准确率：%s'%(i+1,float(y_yuce/y_ture))
    if y_ture!=0.0:
        print '第%s次预测数：%s，真实数为：%s，准确率为：%s'%(i+1,float(y_yuce),y_ture,float(y_yuce/y_ture))
    else:
        print '第%s次预测数：%s，真实数为：%s'%(i+1,float(y_yuce),y_ture)
        




