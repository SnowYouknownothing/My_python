# -*- coding: utf-8 -*-
'''
Created on 2017年12月26日
@author: Administrator
'''
# import tensorflow as tf
# a=tf.constant(0.1, dtype=tf.float32,shape=[3,3])
# b=tf.ones_like(a*100)
# c=tf.zeros_like(a*100)
# with tf.Session() as sess:
#     sess.run(a)
#     print(sess.run(a))
#     print(sess.run(b))
#     print(sess.run(c))
# import numpy as np
# import matplotlib.pyplot as plt
# 
# a=np.linspace(-10, 10, 100)
# b=np.tanh(a)
# c=1/(1+np.exp(-a))
# plt.plot(b,'red')
# plt.plot(c,'green')
# plt.show()
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
# 
# minst=input_data.read_data_sets('minst_data')
# x_train=minst.train.images
# 
# 
# fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
# imgs=np.reshape(x_train[:200],[200,28,28])
# for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
#     for img,ax in zip(image,row):
#         ax.imshow(img)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0.1)
# plt.show()
# list1=[]
# m=0
# for i in range(3):
#     for j in range(550):
#         m+=1
#         if m%165==0:
#             list1.append([i,j])
# 
# 
# 
# list1=np.array(list1)
# print(list1.shape,list1)
# 
# list_g=list(np.random.randint(1,100,[10,100,784]))
# # print(list_g.shape)
# 
# y=np.random.uniform(1,100,20)
# print(y)
# 
# index_list=np.random.randint(1,100,20)
# list_show=[]
# for m in range(10):
#     for n in index_list:
#         list_show.append(list_g[m,n])
# 
# list_show=np.array(list_show)
# print(list_show.shape)


# x3=sorted(np.random.rand(100))
# y=np.linspace(-10, 10,100)
# print(y)
# for i in range(100):
#     if y[i]<0:
#         y[i]=0
# print(y)
# plt.plot(x)

# plt.scatter(x1,y)
# plt.scatter(x2,y)
# plt.scatter(x3,y)

# plt.show()

# x=np.random.uniform(-1,1,1000)
# print(max(x),min(x))
# y=np.linspace(0,0.1,1000)
# plt.plot(x)
# plt.scatter(y, x)
# plt.show()
# for i in range(1,4):
#     print(i)


# a=-1200300
# b=list(str(a))
# print(b)
# c=b[::-1]
# print(c)
# # e=list(map(int,c))
# # print(e)
# while b[-1]=='0':
#     b=b[:-1]
# if b[0]=='-':
#     b=b[1:]
# print(b)
# sum=0
# for i in range(len(b)):
#     sum+=int(b[i])*(10**i)
# print(sum)
# 
# 
# x = np.linspace(start=-10, stop=10, num=101)
# plt.plot(x, np.absolute(x))
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


# path='F:\\BaiduYunDownload\\深度学习代码课件\\对抗生成网络\\卡通图像\\faces\\'
# print(np.array(os.listdir(path)).shape)
# faces_kt=[]
# i=0
# for file in os.listdir(path):
#     images=mpimg.imread(path+file)
#     faces_kt.append(images)
#     i+=1
#     if i==201:
#         break
# faces_kt=np.array(faces_kt)
# print(faces_kt.shape)
# # np.save('faces_kt',faces_kt)
# 
# # images=np.load('faces_kt.npy')
# # print(images.shape)
# fig,axes=plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=(20,10))
# imgs=faces_kt[:200]
# for image,row in zip([imgs[:20],imgs[20:40],imgs[40:60],imgs[60:80],imgs[80:100],imgs[100:120],imgs[120:140],imgs[140:160],imgs[160:180],imgs[180:200]],axes):
#     for img,ax in zip(image,row):
#         ax.imshow(img)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# fig.tight_layout(pad=0.1)
# plt.show()




# x=np.linspace(0,100,21)[:-1]
# print(x,x.shape)
# for i in range(0,6,2):
#     print(i)
# 
# 
# 
# 
# class Solution(object):
# 
#     def isValid(self, s):
#         """
#         :type s: str
#         :rtype: bool
#         """
#         a=b=0
#         list_s=list(s)
#         sum=list_s
#         count1,count2=[],[]
#         list_index=['(',')','[',']','{','}']
#         if len(list_s)%2==0:
#             for i in range(0,len(list_s),2):
#                 for j in range(0,6,2):
#                     if list_index[j]==list_s[i] and list_index[j+1]==list_s[i+1]:
#                         a=1
#                         print(list_index[j],list_s[i],list_index[j+1],list_s[i+1])
#         if len(sum)%2==0:
#             for m in range(int(len(sum)/2)):
#                 count1.append(sum[m])
#             for n in range(int(len(sum)/2),len(sum)):
#                 count2.append(sum[n])
#             for i in range(len(count1)):
#                 if count1[i]=='(':
#                     count1[i]=')'
#                 if count1[i]=='[':
#                     count1[i]=']'
#                 if count1[i]=='{':
#                     count1[i]='}'
#             count1=count1[::-1]
#             print(count1,count2)
#             if count1==count2:
#                 b=1
#         if a ==1 or b==1:
#             return True
#         else:
#             return False
#         
# s=Solution()
# w='[([]])'
# # print(s.isValid(w))
# print('[]' in w)
# 
# w=w.replace('[(','')
# print(w)
# s='{[()]}'
# 
# def isValid(s):
#     if len(s)%2==1:
#         return False
#     else:
#         while ('()' in s) or ('[]' in s) or ('{}' in s):
#             s = s.replace('()','').replace('[]','').replace('{}','')
#         return (s=='')
# 
# print(isValid(w),isValid(s))
# 
# list1=list(w)
# print(list1)
# list1.remove(']')
# print(list1)
# a=[-1,2,3,-0.5]
# print(np.abs(a))

x=np.linspace(-1,1,10)
print(x.shape)
y=x.T
print(y.shape)




