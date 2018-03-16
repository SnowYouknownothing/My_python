# -*- coding: utf-8 -*-
'''
Created on 2018年2月6日
@author: Administrator
'''
import os
import pandas as pd #导入数据分析的利器pandas
import numpy as np
import matplotlib.pyplot as plt
#设置画图时的文字格式为微软雅黑，显示中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.style.use("ggplot")

#从文件夹中读取数据.csv文件
os.chdir("D:\Desktop\kaggle\Titanic_20180202") #切换路径
data=pd.read_csv("train.csv") 
print(data.head(10)) 

print("data's mathmathic describtion：\t")
print(data.describe()) 
print("缺少数据：\n",data.isnull().sum())












