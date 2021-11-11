from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

from six import StringIO
import pydot

# load the dataset
iris = load_iris()

print(f"feature names: {iris.feature_names}, "
      "target names: {iris.target_names}")

print(f"sample feature: {iris.data[0]}, target: {iris.target[0]}")


# testing data
# should be separate from the training data

# the dataset is ordered so that
# 0-49 - setosa
# 50-99 - versicolor
# 100 onwards - virginica
# one from each type will be set aside as the testing data
test_idx = [0, 50, 100]

# filter out the training data
train_data = np.delete(iris.data, test_idx, axis=0)
train_targets = np.delete(iris.target, test_idx)

# filter out the test data
test_data = iris.data[test_idx]
test_targets = iris.target[test_idx]

# create classifier
clf = tree.DecisionTreeClassifier()

# train the classifier
clf = clf.fit(train_data, train_targets)

print(f"test targets: {test_targets}")
print(f"prediction tests: {clf.predict(test_data)}")

# visualization
# pip3 install six graphviz pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")
