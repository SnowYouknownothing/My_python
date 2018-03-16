'''

@author: Administrator
'''
import numpy 
from numpy import exp, array, random, dot
from __builtin__ import int
# y=[[1],[2],[2],[2]]
# print y
# x=numpy.array(y)
# print x
# # y=y.T
# # print y
# print x
# x = numpy.array([[1],[2],[2],[2]])
# print x
# x=x.T
# print x
# x=x.dot([[1],[2],[2],[2]])
# print x
y = numpy.array([[1],[2],[2]]).T
print y
x=random.random((3, 1)) 
print x
z=dot(y,x)

print type(z)
w=float(z)
print w






