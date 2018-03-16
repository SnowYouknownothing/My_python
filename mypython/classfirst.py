# -*- coding: utf-8 -*-
'''
Created on 2017年11月17日
@author: Administrator
'''
from numpy import exp, array, random, dot
from cryptography.hazmat.primitives.asymmetric.padding import PSS
class FirstClass:
    color ='green'
    age=26
    judge='True'
    def function(self):
        print 'firstclass'
tt=FirstClass()
c=tt.function()
print tt.age   
# random.seed(1)   
list1_ddd =10*random.random((1, 5))
listl=list(list1_ddd)
list2=[1,2,3,4]
list2.append('1')
print list2

class Mylist(list):
    pass
list3=Mylist()
list3.append(1)
list3.append(3)
list3.append(0)
list3.append(-2)
print list3
list3.sort()
print list3
print '\n'
class Ball:
    def setName(self,name):
        self.name=name
        print self.name
    def click(self):
        print '我叫%s,谁在踢我' % self.name
a=Ball()
b=Ball()
a.setName('球A')
b.setName('球B')
a.click()
b.click()
x=FirstClass()
print x.age




