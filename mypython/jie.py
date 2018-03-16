# -*- coding: utf-8 -*-
'''
Created on 2017年12月12日
@author: Administrator
'''
import jieba
word=jieba.cut('那些教你拍案叫绝的Python库')
print('$'.join(word))

import wget
url=''
out=''