# -*- coding: utf-8 -*-
'''
Created on 2017年11月23日
@author: Administrator
'''
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0, 10, 1000)
y1 = np.sin(x1)
z1 = np.cos(x1**2)
# print x1
plt.figure(figsize=(10,10))
plt.plot(x1,z1,label="$sin(x)$",color="red",linewidth=1)
# plt.plot(x1,z1,"b--",label="$cos(x^2)$")
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")
plt.ylim(-1.2,1.2)
# plt.legend()
plt.show()