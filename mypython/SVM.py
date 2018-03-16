# -*- coding: utf-8 -*-
'''
Created on 2018年1月15日
@author: Administrator
'''
from sklearn import svm
import numpy as np
x=np.array([[2,0],[1,1],2,3])
y=np.array([0,0,1])
clf=svm.SVC(kernel='linear')
clf.fit(x,y)
print(clf)
print(clf.supprot_vectors_)
print(clf.support_)
print(clf.n_support)