# -*- coding: utf-8 -*-
'''
Created on 2018年2月2日
@author: Administrator
'''
import csv
import numpy as np
from sklearn import svm


def get_data(path):
    with open(path) as fn:
        readers=csv.reader(fn)
        rows=[row for row in readers]
    rows.pop(0) 
    return np.array(rows)

def trans_data(train_data):
    x_train_1=np.reshape(train_data[:,-10],[len(train_data),1])
    x_train_2=train_data[:,-8:-4]
    x_train_3=np.reshape(train_data[:,-3],[len(train_data),1])
    x_train_4=np.reshape(train_data[:,-1],[len(train_data),1])
    
    x_train=np.hstack((x_train_1,x_train_2))
    x_train=np.hstack((x_train,x_train_3))
    x_train=np.hstack((x_train,x_train_4))
    for i in range(len(x_train)):
        if x_train[i,1]=='female':
            x_train[i,1]=1
        else:
            x_train[i,1]=0
        
        if x_train[i,-1]=='C':
            x_train[i,-1]=0
        if x_train[i,-1]=='Q':
            x_train[i,-1]=1
        if x_train[i,-1]=='S':
            x_train[i,-1]=2
        if x_train[i,2]=='':
    #         x_train[i,2]=np.random.randint(10,60)
            x_train[i,2]=30
    x_train_=[]
    for i in range(len(x_train)):
        c=x_train[i]
        x_1=[]
        for j in range(len(x_train[0])):
            if '.' in c[j]:
                c[j]=c[j][:(c[j].index('.'))]
            if c[j]=='':
                c[j]=0
            x_1.append(int(c[j]))
        x_train_.append(x_1)
    x_train_=np.array(x_train_)
    return x_train_

    
path_train='D:\\Desktop\\kaggle\\Titanic_20180202\\train.csv'
path_test='D:\\Desktop\\kaggle\\Titanic_20180202\\test.csv'

train_data=get_data(path_train)
test_data=get_data(path_test)
print(train_data.shape,test_data.shape)

x_train_=trans_data(train_data)
x_test=trans_data(test_data)
y_train=np.reshape(train_data[:,1],[891,1])
'''
# print(test_data[1],train_data[1])
# for i in range(len(train_data)):
#     train_data.remove(i,3)

x_train_1=np.reshape(train_data[:,2],[891,1])
x_train_2=train_data[:,4:8]
x_train_3=np.reshape(train_data[:,-3],[891,1])
x_train_4=np.reshape(train_data[:,-1],[891,1])

x_train=np.hstack((x_train_1,x_train_2))
x_train=np.hstack((x_train,x_train_3))
x_train=np.hstack((x_train,x_train_4))

y_train=np.reshape(train_data[:,1],[891,1])


print(x_train.shape,x_train_1.shape,x_train_2.shape,y_train.shape)

print(x_train[1],y_train[1])

print('step1')

for i in range(len(x_train)):
    if x_train[i,1]=='female':
        x_train[i,1]=0
    else:
        x_train[i,1]=1
    
    if x_train[i,-1]=='C':
        x_train[i,-1]=0
    if x_train[i,-1]=='Q':
        x_train[i,-1]=1
    if x_train[i,-1]=='S':
        x_train[i,-1]=2
    if x_train[i,2]=='':
#         x_train[i,2]=np.random.randint(10,60)
        x_train[i,2]=30
print(x_train[5],y_train[5])
print(x_train[6],y_train[6])

print('step2')


x_train_=[]
y_train_=np.zeros([len(y_train),1])
for i in range(len(x_train)):
    if str(y_train[i])=="['0']":
        y_train_[i]=0
    else:
        y_train_[i]=1
    c=x_train[i]
    x_1=[]
    for j in range(len(x_train[0])):
        if '.' in c[j]:
            c[j]=c[j][:(c[j].index('.'))]
        if c[j]=='':
            c[j]=0
        x_1.append(int(c[j]))
    x_train_.append(x_1)
x_train_=np.array(x_train_)

'''
y_train_=np.zeros([len(y_train),1])
for i in range(len(x_train_)):
    if str(y_train[i])=="['0']":
        y_train_[i]=0
    else:
        y_train_[i]=1


print(x_train_.shape,x_train_[5])
print(x_test.shape,x_test[5])
print(y_train_.shape,y_train_[1])


from sklearn import svm
# x=np.array([[2,0],[1,1],2,3])
# y=np.array([0,0,1])


clf=svm.SVC(kernel='linear')
# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train_,y_train_)
# print(clf)
# print(clf.supprot_vectors_)
# print(clf.support_)
# print(clf.n_support)
# prediction_=clf.predict(x_train_[850:])
# print(prediction_.shape)
# prediction_=np.reshape(prediction_,[len(prediction_),1])
# acc=y_train_[850:]-prediction_
# # print(acc)
# print(acc.shape)
# sum1=0
# for i in range(len(acc)):
#     if acc[i]==0.:
#         sum1+=1
# print(sum1/(len(acc)))

# print(prediction_)
prediction_1=clf.predict(x_test)
print(prediction_1.shape)
with open('D:\\Desktop\\kaggle\\Titanic_20180202\\y_test.csv','w',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['PassengerId','Survived'])
    for i in range(len(prediction_1)):
        data=(i+892,prediction_1[i])
        writer.writerow(data)



print('done!')