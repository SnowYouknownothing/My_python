'''
@author: Administrator
'''
from sklearn import neighbors
from sklearn import datasets

knn=neighbors.KNeighborsClassifier()

iris= datasets.load_iris()

print iris

knn.fit(iris.data, iris.target)
# print(knn)

predictedLabel=knn.predict([[10,4,5,2]])

print predictedLabel



