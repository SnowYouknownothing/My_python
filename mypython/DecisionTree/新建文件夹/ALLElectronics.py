#coding:utf-8
#author:snow
# from sklearn.feature_extraction import DictVectorizer
# import csv 
# from sklearn import preprocessing
# from sklearn import tree
# from sklearn.externals.six import StringIO
# from blaze.expr.expressions import label
#     
# # read
# tree1 = open(r'D:\DLdata\Tree5.csv', 'rb' )
# reader = csv.reader(tree1)
# # print(reader)
# headers = next(reader)
# print headers
# 
# featureList = []
# labelList = []
# for row in reader:
#     labelList.append(row[len(row)-1])
#     rowDict = {}
#     for i in range(1,len(row)-1):
#         rowDict[headers[i]]=row[i]
#     featureList.append(rowDict)
# print featureList
# 
# vec = DictVectorizer
# dummyX = vec.fit_transform(featureList) .toarry()
#  
# print 'dummyX' + str(labelList)
# print vec.get_feature_names()
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO
from xml.sax.handler import feature_external_ges
from numpy.distutils.fcompiler import dummy_fortran_file

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'D:\DLdata\Tree5.csv', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)
featureList = []
lableList = []
for row in reader:
    lableList.append(row[len(row)-1])
    rowDict = {}
#不包括len(row)-1
for i in range(1,len(row)-1):
    rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)

vec = DictVectorizer()
dummX = vec.fit_transform(featureList).toarray()
print(str(dummX))
lb = preprocessing.LabelBinarizer()
dummY = lb.fit_transform(lableList)
print(str(dummY))

#entropy=>ID3
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummX, dummY)
print("clf:"+str(clf))


#可视化tree
# with open("resultTree.dot",'w')as f:
# f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(),out_file = f)
# 
# 
# #对于新的数据怎样来查看它的分类
# oneRowX = dummX[0,:]
# print("oneRowX: "+str(oneRowX))
# newRowX = oneRowX
# newRowX[0] = 1
# newRowX[2] = 0
# 
# predictedY = clf.predict(newRowX)
# print("predictedY: "+ str(predictedY))