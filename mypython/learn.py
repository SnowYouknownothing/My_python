# # -*- coding: utf-8 -*-
# '''
# Created on 2017年11月16日
# @author: Administrator
# '''
# def MyFirstFunction():
#     'dfgerfger'
#     print '这是我创建的第一个函数，我表示很激动'
#     print '在此我要感谢CCTV，MTV'
# MyFirstFunction()
# 
# def MySecondFunction(a,b,c):
#     return a+b+c
#     f1
# 
# e=MySecondFunction(1,2,3)
# print e
# for i in range(10):
#     print MySecondFunction(i,i,i)
# def MyThreeFunction(a,b,c):
#     return a*a,b*b,c*c
# for i in range(10):
#     print MyThreeFunction(i,i,i)
# count=10
# def f1():
#     global count
#     count=1
#     print count
#     for i in range(5):
#         print MyThreeFunction(1, i, 2)
# def funx(x):
#     def funy(y):
#         return x*y
#     return funy
# a=funx(8)
# print type(a)
# print a(2)
# g=lambda x:2*x-33
# print g(2)
#   
# help(filter)
# 
# 
# 
# 
# 
# 
# 
# 
# print list(filter(None, [1,0,False,True]))
# print list(map(lambda x: x*2, range(10)))

# f=open('G:\\CATIA V5-6R2013 64位 简体中文破解版.txt')
# f.read()
# 
# 
# 
# f.close()
# f=open('E:\\first.txt','w')
# f.write('11111')
# f.close()
class Rectangle:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def getPeri(self):
        return (self.x+self.y)*2
rect=Rectangle(3,4)

print rect.getPeri()
    
class CapStr(str):
    def __new__(cls, string):
        string=string.upper()
        return str.__new__(cls,string)
a= CapStr('dddddd')
print a

class int(int):
    def __add__(self,other):
        return int.__sub__(self,other)
a=int('11')
b=int('22')
print a+b

print 1+2

a={'xxxxxxxxxxxxxx':'dddddddddddd',\
   'ddddddddd':22222,'dddd':3333,\
   'ddd':333}
print a
for i in a:
    print i,a[i]

# class Fibs:
#     def __init__(self):
#         self.x=0
#         self.y=1
#     def 
#     
    
# a=0
# b=1
# for i in range(1,10):
#     c=a+b 
#     print a,'\n',b,'\n',c
#     a=b+c
#     b=c+a
# hjgf=a/b
# print hjgf
def myGen():
    print'dddd'
    yield 1
    yield 2
    
mygen = myGen()
next(mygen)



