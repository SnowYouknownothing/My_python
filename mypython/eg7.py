# -*- coding: utf-8 -*-
'''
Created on 2018年1月9日
@author: Administrator
'''
'''
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data
x_train,y_train=[],[]
for i in range(1,6):
    datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//data_batch_%s'%(i))
    for j in range(len(datadict[b'data'])):
        x_train.append(datadict[b'data'][j])
        y_train.append(datadict[b'labels'][j])
# print(len(x_test),len(x_test[0]),len(y_test))

datadict=unpickle('D://Users//Administrator.lxp-PC//eclipse-workspace//AN3//cifar-10-batches-py//test_batch')
x_test,y_test=datadict[b'data'],datadict[b'labels']
print(len(x_test),len(x_test[0]),len(y_test))
print(len(x_train),len(y_train))
#

x_list=[]
for i in range(len(y_train)):
    if y_train[i]==1:
        x_list.append(x_train[i])

# x_real=((np.array(x_list))/127.5)-1



x_list=np.transpose((np.reshape(np.array(x_list),[5000,3,32,32])),[0,2,3,1])
print(x_list.shape)



fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=x_list[:200]
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()
'''

# import numpy as np
# a=np.random.randint(1,10,[2,5] )
# b=np.random.randint(1,10,[3,5] )
# print(a)
# print(b)
# c=np.concatenate((a,b),axis=0)
# print(c)

'''
import csv
csvfile=open('D:\\Desktop\\kaggle\\digits_20180122\\kaggle_test.csv','w')
writer=csv.writer(csvfile)
data0=['imageld','label']
data1=[1,2]
data2=[2,3]
data3=[2,3]
writer.writerows([data1,data2,data3])
# writer.writerow(data2)
csvfile.close()
'''
# import csv
# with open("D:\\Desktop\\kaggle\\digits_20180122\\test1.csv","w",newline ='') as csvfile: 
#     writer = csv.writer(csvfile)
#     writer.writerow(["index","a_name","b_name"])
#     writer.writerows([[0,1,3],[1,2,3],[2,3,4]])


import csv 
import numpy as np
import matplotlib.pyplot as plt


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
    x_train=np.array(x_train)
    print(x_train.shape)


#     print(x_labels[1:5],x_labels_[1:5])
x_images=np.reshape(x_train,[len(x_train),28,28])

# plt.imshow(x_images[0])
# plt.show()

import scipy.misc
scipy.misc.imsave('D:\\Desktop\\kaggle\\digits_20180122\\12.jpg',x_images[0])
from PIL import Image
im=Image.open('D:\\Desktop\\kaggle\\digits_20180122\\12.jpg')
# im.show()
im_rotate=im.rotate(30)
# im_rotate.show()
im_rotate.save('D:\\Desktop\\kaggle\\digits_20180122\\13.jpg')

y=scipy.misc.imread('D:\\Desktop\\kaggle\\digits_20180122\\13.jpg')

plt.imshow(x_images[0])
plt.show()
plt.imshow(y)
plt.show()


'''

fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=x_images[:200]
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()

print(x_images.shape)
for i in range(len(x_images)):
    
    x_images[i]=x_images[i].T
print(x_images.shape)

fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
imgs=x_images[:200]
for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
    for img,ax in zip(image,row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()

# from PIL import Image
# 
# 
# im=Image.open('D:\\Desktop\\kaggle\\digits_20180122\\1.jpeg')
# im.show()
# 
# 
# 
# print(type(im))
# 
# im_rotate=im.rotate(45)
# im_rotate.show()

'''






