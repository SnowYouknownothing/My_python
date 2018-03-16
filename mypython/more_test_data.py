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



path='D:\\Desktop\\kaggle\\digits_20180122\\test.csv'

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
x_images=np.array(x_train)
print(x_images.shape)

x_images=np.reshape(x_images,[len(x_images),28,28])
new_images_1_list,new_images_2_list=[],[]

for i in range(len(x_images)):
    scipy.misc.imsave('D:\\Desktop\\kaggle\\digits_20180122\\more_test_data1\\%s.jpg'%(i),x_images[i])
    im=Image.open('D:\\Desktop\\kaggle\\digits_20180122\\more_test_data1\\%s.jpg'%(i))
    im_rotate=im.rotate(15)
    im_rotate.save('D:\\Desktop\\kaggle\\digits_20180122\\more_test_data1\\n%s.jpg'%(i))
    new_images_1=scipy.misc.imread('D:\\Desktop\\kaggle\\digits_20180122\\more_test_data1\\n%s.jpg'%(i))
    new_images_1_list.append(new_images_1)
    im_rotate=im.rotate(-15)
    im_rotate.save('D:\\Desktop\\kaggle\\digits_20180122\\more_test_data1\\m%s.jpg'%(i))
    new_images_2=scipy.misc.imread('D:\\Desktop\\kaggle\\digits_20180122\\more_test_data1\\m%s.jpg'%(i))
    new_images_2_list.append(new_images_2)

new_images_1_list=np.reshape(np.array(new_images_1_list),[len(new_images_1_list),28*28])
new_images_2_list=np.reshape(np.array(new_images_2_list),[len(new_images_2_list),28*28])
print(new_images_1_list.shape,new_images_2_list.shape)

np.save('D:\\Desktop\\kaggle\\digits_20180122\\more_data1_py\\new_test_data1.npy',new_images_1_list)
np.save('D:\\Desktop\\kaggle\\digits_20180122\\more_data1_py\\new_test_data2.npy',new_images_2_list)

print('done!')
