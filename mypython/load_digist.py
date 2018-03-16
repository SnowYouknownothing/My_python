# -*- coding: utf-8 -*-
'''
Created on 2017年11月30日
@author: Administrator
'''
from sklearn.datasets import load_digits
def sklearn_data():
    digits=load_digits()
    x=digits.data
    y=digits.target
    return x,y
    