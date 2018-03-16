# -*- coding: utf-8 -*-
'''
Created on 2017年12月12日
@author: Administrator
'''

import numpy as np
import matplotlib.pyplot as plt

def sample_data(size):
    data=[]
    for _ in range(size):
        data.append(np.random.normal(1,5))
    return np.array(data)

eg1=sample_data(1000)
plt.figure()
plt.plot(eg1)
plt.show()

def random_data(size,length=100):
    data=[]
    x=np.random.random(length)
    data.append(x)
    return np.array(data)

def preprocess_data(x):return [[np.mean(data),np.std(data)]for data in x]

























