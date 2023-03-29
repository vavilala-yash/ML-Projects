import numpy as np
import pandas as pd
iris = pd.read_csv("C:\\Users\\V YASWANTH SAI\\PycharmProjects\\pythonProject\\iris.csv")
print(iris)
a = iris.drop(columns=['class'])
b = iris['class']
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.4)
from sklearn import neighbors
classifier = neighbors.KNeighborsClassifier()
classifier.fit(a_train, b_train)
predictions = classifier.predict(a_test)
from sklearn.metrics import accuracy_score
acc1 = (accuracy_score(b_test, predictions)*100)
print(acc1)
from sklearn import tree
dstreeclassifier = tree.DecisionTreeClassifier()
dstreeclassifier.fit(a_train, b_train)
dstreepredictions=dstreeclassifier.predict(a_test)
acc2 = (accuracy_score(b_test, dstreepredictions)*100)
print(acc2)
import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle("Comparison of Algorithms")
names = ['KNN', 'Decison Tree']
result = [acc1, acc2]
plt.bar(names, result)
plt.xlabel("Algorithm")
plt.ylabel("Acccuracy")
plt.show()
