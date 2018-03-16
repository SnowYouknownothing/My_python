# -*- coding: utf-8 -*-
'''
Created on 2018年1月2日
@author: Administrator
'''
import numpy as np
from math import sin
import matplotlib.pyplot as plt

x=np.linspace(0,1,100)

y1=(6*x+np.sqrt(36*(x**2)-20*(5*(x**2)-128)))/10
y2=(6*x-np.sqrt(36*(x**2)-20*(5*(x**2)-128)))/10

x2=np.linspace(-1,0,100)
y3=(-6*x+np.sqrt(36*(x**2)-20*(5*(x**2)-128)))/10
y4=(-6*x-np.sqrt(36*(x**2)-20*(5*(x**2)-128)))/10

plt.plot(y1)
plt.plot(y2)
plt.plot(y3)
plt.plot(y4)
plt.show()

