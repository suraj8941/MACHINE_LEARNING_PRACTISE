# LOGISTIC REGRESSION SAMPLE CODE

from sklearn import datasets
import numpy as np
from sklearn.linear_model  import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()

X = iris["data"][:, 3: ]
Y = (iris["target"] == 2).astype(np.int)
clf = LogisticRegression()
clf.fit(X, Y)
example = clf.predict(([[2.6]]))
print(example)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
Y_prob = clf.predict_proba(X_new)
plt.plot(X_new, Y_prob[:, 1], "g-", Label="virginica")
plt.show()
#print(list(iris.keys()))
#print(iris['target'])