# -*- coding: utf-8 -*-
'''
Created on 2018年3月6日
@author: Administrator
'''

'''
0000001.jpg,trousers,430_284_0,713_303_0,560_537_1,560_626_1,361_588_1,573_622_1,-1_-1_-1
0000002.jpg,trousers,359_301_1,464_297_1,417_403_1,340_669_1,308_658_1,456_713_1,491_714_1 

0000001.jpg,trousers,430_294_0,713_323_0,560_567_1,560_666_1,361_638_1,573_682_1,123_345_1
0000002.jpg,trousers,359_311_1,464_317_1,417_433_1,340_709_1,308_708_1,456_773_1,491_784_1

0000001.jpg,0,0,0.105, 0.141, 0.176, 0.212,0
0000002.jpg,0.095, 0.190, 0.285, 0.381, 0.476, 0.570,0.666

NE=30.00%

取两裤头的欧式距离为归一化参数。在本示例中，对于第一张图，归一化参数为283.64，第二张图的归一化参数为105.08。

'''
import numpy as np
def pre_deal(x):
    list_num=[]
    m=0
    while m!=-1:
        m=x.find(',')
        list_num.append(x[:m])
        x=x[m+1:]
    list_num[-1]=list_num[-1]+x[-1]

    list_int=[]
    for i in range(len(list_num)):
        x_1=list_num[i]
        n=0
        while n!=-1:
            n=x_1.find('_')
            list_int.append(x_1[:n])
            x_1=x_1[n+1:]
        list_int[-1]=list_int[-1]+x_1[-1]
    return np.array([int(x) for x in list_int])

str_1_test='430_284_0,713_303_0,560_537_1,560_626_1,361_588_1,573_622_1,-1_-1_-1'
str_2_test='359_301_1,464_297_1,417_403_1,340_669_1,308_658_1,456_713_1,491_714_1'
str_1_train='430_294_0,713_323_0,560_567_1,560_666_1,361_638_1,573_682_1,123_345_1'
str_2_train='359_311_1,464_317_1,417_433_1,340_709_1,308_708_1,456_773_1,491_784_1'

x_1_test=pre_deal(str_1_test)
x_2_test=pre_deal(str_2_test)
x_1_train=pre_deal(str_1_train)
x_2_train=pre_deal(str_2_train)

def x_y_show_hide(x,num):
    return np.array([out for out in x[range(num,len(x),3)]])
def distance(x,y):
    return np.sqrt(x**2+y**2)

x_1_test_x,x_1_test_y,x_1_test_show_hide=x_y_show_hide(x_1_test,0),x_y_show_hide(x_1_test,1),x_y_show_hide(x_1_test,2)
x_2_test_x,x_2_test_y,x_2_test_show_hide=x_y_show_hide(x_2_test,0),x_y_show_hide(x_2_test,1),x_y_show_hide(x_2_test,2)
x_1_train_x,x_1_train_y,x_1_train_show_hide=x_y_show_hide(x_1_train,0),x_y_show_hide(x_1_train,1),x_y_show_hide(x_1_train,2)
x_2_train_x,x_2_train_y,x_2_train_show_hide=x_y_show_hide(x_2_train,0),x_y_show_hide(x_2_train,1),x_y_show_hide(x_2_train,2)
print(x_1_test_x,x_1_test_y,x_1_test_show_hide)
print(x_2_test_x,x_2_test_y,x_2_test_show_hide)
print(x_1_train_x,x_1_train_y,x_1_train_show_hide)
print(x_2_train_x,x_2_train_y,x_2_train_show_hide)

x_1_distance=np.sqrt((x_1_test_x-x_1_train_x)**2+(x_1_test_y-x_1_train_y)**2)
print(x_1_distance)

# 430_284_0,713_303_0
x_1_middle=np.sqrt((430-713)**2+(284-303)**2)
x_2_middle=np.sqrt((359-464)**2+(301-297)**2)

x_jug=x_1_distance/x_1_middle
print(x_jug)
print(np.sum(x_jug[:-1])/6)

print((0.105+0.141+0.176+0.212+0.095+0.190+0.285+0.381+0.476+0.570+0.666)/11)







