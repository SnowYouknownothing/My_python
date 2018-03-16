# -*- coding: utf-8 -*-
'''
Created on 2018年1月5日
@author: Administrator
'''
# file=F:\BaiduYunDownload\深度学习代码课件\对抗生成网络\卡通图像
from os import walk

import matplotlib.pyplot as plt


('F:\BaiduYunDownload\深度学习代码课件\对抗生成网络\卡通图像\faces\0000fdee4208b8b7e12074c920bc6166-0.jpg')



def read_images(path):
    filenames=next(walk(path))[2]

    return filenames

path='F:\BaiduYunDownload\深度学习代码课件\对抗生成网络\卡通图像'

print(read_images(path))
    
