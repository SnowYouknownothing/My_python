'''

@author: Administrator
'''
from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
print training_set_inputs
training_set_outputs = array([[0, 1, 1, 0]]).T
print training_set_outputs
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
print synaptic_weights
print '\n'
for iteration in xrange(100):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
#     print output
#     print '\n'
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
#     print synaptic_weights
#     print '\n'
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
print synaptic_weights

# print array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# print ([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# print array([[0, 1, 1, 0]]).T
# print ([[0, 1, 1, 0]]).T
# print 2*random.random((3, 1))
# a= 2 * random.random((3, 1)) - 1
# print a
# print synaptic_weights
# a=array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# print a[0,0],a[0,1],a[0,2],a[1,0]
# def sigmord(x):
#     return 1/(1+exp(-x))
# def error(x,y):
#     return x*(1-x)*(y-x) 
# a=array([1,2,3])
# b=array([4,5,6])
# print sigmord(a)
# print error(a, b)
# for i in range(1,4):
#     print 1/(1+exp(-i))
