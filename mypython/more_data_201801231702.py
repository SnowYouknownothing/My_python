# -*- coding: utf-8 -*-
'''
Created on 2018年1月23日
@author: Administrator
'''
import csv 
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image



path='D:\\Desktop\\kaggle\\digits_20180122\\train.csv'

with open(path,encoding='utf-8') as fn:
    reader=csv.reader(fn)
    rows=[row for row in reader]
rows.pop(0)
x_train=[]
for i in range(len(rows)):
    c=rows[i]
    x_1=[]
    for j in range(len(rows[0])):
        x_1.append(int(c[j]))
    x_train.append(x_1)
x_train=np.array(x_train)
print(x_train.shape)

x_images=x_train[:,1:]
x_labels=x_train[:,0]
print(x_images.shape,x_labels.shape)
print(x_images.shape)
print(np.max(x_images),np.min(x_images))
x_labels_=np.zeros([len(x_labels),10])
for i in range(len(x_labels)):
    x_labels_[i,x_labels[i]]=1

x_images=np.reshape(x_images,[len(x_images),28,28])
new_images_1_list,new_images_2_list=[],[]

save_data=True

if save_data==True:
    for i in range(len(x_images)):
        scipy.misc.imsave('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\%s.jpg'%(i),x_images[i])
        im=Image.open('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\%s.jpg'%(i))
        im_rotate=im.rotate(15)
        im_rotate.save('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\n%s.jpg'%(i))
        new_images_1=scipy.misc.imread('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\n%s.jpg'%(i))
        new_images_1_list.append(new_images_1)
        im_rotate=im.rotate(-15)
        im_rotate.save('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\m%s.jpg'%(i))
        new_images_2=scipy.misc.imread('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\m%s.jpg'%(i))
        new_images_2_list.append(new_images_2)
else:
    for i in range(len(x_images)):
        new_images_1=scipy.misc.imread('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\n%s.jpg'%(i))
        new_images_1_list.append(new_images_1)
        new_images_2=scipy.misc.imread('D:\\Desktop\\kaggle\\digits_20180122\\more_data1\\m%s.jpg'%(i))
        new_images_2_list.append(new_images_2)

new_images_1_list=np.reshape(np.array(new_images_1_list),[len(new_images_1_list),28*28])
new_images_2_list=np.reshape(np.array(new_images_2_list),[len(new_images_2_list),28*28])
print(new_images_1_list.shape,new_images_2_list.shape)

np.save('D:\\Desktop\\kaggle\\digits_20180122\\more_data1_py\\new_data1.npy',new_images_1_list)
np.save('D:\\Desktop\\kaggle\\digits_20180122\\more_data1_py\\new_data2.npy',new_images_2_list)

print('done!')




