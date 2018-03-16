 # -*- coding: utf-8 -*-
'''
Created on 2017年11月17日
@author: Administrator
'''

class Ball:
    def __init__(self,name):
        self.name=name
    def kick(self):
        print '我是%s,该死的，谁踢我'%self.name
a=Ball('土豆')
b=Ball('球A')
a.kick()
b.kick()
class FirstClass:
    color ='green'
    __age=26
    judge='True'
    def function(self):
        print 'firstclass'
c=FirstClass()
c.function()
print c._FirstClass__age
class SecondClass(FirstClass):
    color = 'black'
#     print a
#     pass
e=SecondClass()
print e.color 
print e.judge
print issubclass(SecondClass, FirstClass)
print issubclass(SecondClass, FirstClass)
print issubclass(SecondClass, FirstClass)


