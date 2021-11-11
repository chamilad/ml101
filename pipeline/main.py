from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()

# f(x) = y
x = iris.data
y = iris.target

# split to train and test data equally
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)


# create and train classifier_1
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(x_train, y_train)

# create and train class_2
clf2 = KNeighborsClassifier()
clf2 = clf2.fit(x_train, y_train)

prd1 = clf1.predict(x_test)
print(f"accuracy dtc: {accuracy_score(y_test, prd1)}")

prd2 = clf2.predict(x_test)
print(f"accuracy knc: {accuracy_score(y_test, prd2)}")
