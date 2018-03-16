# -*- coding: utf-8 -*-
'''
Created on 2018年1月17日
@author: Administrator
'''
import matplotlib.pyplot as plt
import cv2

print(cv2.__version__)

print('hello,world')
img=cv2.imread('F:\\BaiduYunDownload\\深度学习代码课件\\对抗生成网络\\卡通图像\\0000fdee4208b8b7e12074c920bc6166-0.JPG')
print(type(img))
plt.imshow(img)
plt.show()
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()