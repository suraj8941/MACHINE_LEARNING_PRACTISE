# K nearest neighbour program ...........

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


# Loading datatsets............
iris = datasets.load_iris()

# Description about data.......
print(iris.DESCR)
features = iris.data
labels = iris.target

# Training the classifier......
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[8, 9, 7, 9]])
print(preds)