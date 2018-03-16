# -*- coding: utf-8 -*-
'''
Created on 2018年3月9日
@author: Administrator
'''
import numpy as np
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt

x_data_path ='D:\\Desktop\\天池\\real_train\\train\\Images\\'
y_data_path = 'D:\\Desktop\\天池\\real_train\\train\\Annotations\\train.csv'

x_data_list_name=os.listdir(x_data_path)
print(x_data_list_name,y_data_path)
x_data_list,y_data_list=[],[]
sum_x=0
for i in range(len(x_data_list_name)):
    list_data=os.listdir(x_data_path+x_data_list_name[i])
    print(len(list_data))
    for j in range(len(list_data)):
        x_data_list.append(list_data[j])
    sum_x+=len(list_data)
print(len(x_data_list))
print(x_data_list[0])
print(sum_x)

im=plt.imread(x_data_path+x_data_list_name[0]+'\\'+x_data_list[0])
print(im.shape)
# plt.imshow(im)
# plt.show()

with open(y_data_path,encoding='utf-8') as fn:
    readers=csv.reader(fn)
    rows=[row for row in readers]
rows.pop(0)
print(len(rows))
print(rows[0])
print(len(rows[0]))


def pre_deal(x):
    list_int=[]
    for i in range(len(x)):
        x_1=x[i]
        n=0
        while n!=-1:
            n=x_1.find('_')
            list_int.append(x_1[:n])
            x_1=x_1[n+1:]
        list_int[-1]=list_int[-1]+x_1[-1]
    return np.array([int(x) for x in list_int])

# print(pre_deal(rows[2:]))
x_test=rows[0]
print(x_test[2:])
x_test=pre_deal(x_test[2:])
print(x_test)
print(x_test.shape)

