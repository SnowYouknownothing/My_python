# -*- coding: utf-8 -*-
'''
Created on 2018年1月29日
@author: Administrator
'''


import numpy as np
import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
'''
# a=np.random.random([3,3])
# b=np.random.random([3,1])
# print(a,b)
# c=np.hstack((a,b))
# print(c.shape)

c='7.25'
x=c.index('.')
print(x)
x=int(c[:x])    
print(x)

print(np.random.randint(10,15))
a=np.random.random((10,10))
print(a)
c=np.round(a)
print(c)

print(np.zeros((10,2)))

'''

'''
im =Image.open('D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\77629b9e3.png')
out=im.resize((1024,1024))
out.save('D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\22.png')
images_1=np.array(mpimg.imread('D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\77629b9e3.png'))
print(images_1.shape)
plt.imshow(images_1)
plt.show()
images_2=np.array(mpimg.imread('D:\\Desktop\\kaggle\\Plant_Seedlings_Classification\\22.png'))
print(images_2.shape)
plt.imshow(images_2)
plt.show()
c=np.zeros((1024, 1024, 3))
# for i in range(1024):
#     for j in range(1024):
#         for m in range(3):
#             c[i,j,m]=images_2[i,j,m]
c=images_2[:,:,:3]
print(c.shape)
plt.imshow(c)
plt.show()

print('done!')




# a=np.ones((3,2,2,2))
# b=np.ones((5,2,2,2))
# c=np.vstack((a,b))
# print(c.shape)
a=np.ones(10)
b=np.random.random((1,10))
print(a.shape,b.shape)
c=b+a
print(c.shape)

print(a,b,c)

a=np.random.random((100,100,3))
plt.imshow(a)
plt.show()



'''


f=open(r'C:\\Users\\Administrator.lxp-PC\\.ssh\\id_rsa.pub','r')
line=f.readline()
while line:
    print(line)
    line=f.readline()
























