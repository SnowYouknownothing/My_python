# -*- coding: utf-8 -*-
'''
Created on 2018年1月8日
@author: Administrator
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

load_datas=True
while load_datas==True:
    path='F:\\BaiduYunDownload\\深度学习代码课件\\对抗生成网络\\卡通图像\\faces\\'
    print(np.array(os.listdir(path)).shape)
    faces_kt=[]
    for file in os.listdir(path):
        images=mpimg.imread(path+file)
        faces_kt.append(images)
    faces_kt=np.array(faces_kt)
    faces_kt=faces_kt/127.5-1
    print(faces_kt.shape)
    np.save('faces_kt1',faces_kt)
    print('save data successful!')
    load_datas=False

images=np.load('faces_kt1.npy')
print(images.shape,np.max(images),np.min(images),'done')
# fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
# imgs=images[:200]
# for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
#     for img,ax in zip(image,row):
#         ax.imshow(img)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0.1)
# plt.show()























