from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from scipy.spatial import distance


def euc(a, b):
    """
    return the euclidean distance between a and b
    """
    return distance.euclidean(a, b)


class ScrappyKNN():
    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

        print(f"trainset size: {len(self.xtrain)}, "
              f"trainlabel size: {len(self.ytrain)}")

    def predict(self, xtest):
        predictions = []
        for row in xtest:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        """
        k1n
        find the closest neighbor from the training data set
        """

        best_dist = euc(row, self.xtrain[0])
        best_dist_idx = 0

        for i in range(1, len(self.xtrain)):
            dist = euc(row, self.xtrain[i])

            if dist < best_dist:
                best_dist = dist
                best_dist_idx = i

        # return the label, f(x) = y
        return self.ytrain[best_dist_idx]


# load data set
iris = datasets.load_iris()

x = iris.data
y = iris.target

# split train test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

# create and train model
clf = ScrappyKNN()
clf.fit(xtrain, ytrain)

# test a prediction
prds = clf.predict(xtest)
print(f"accuracy (scrappy_knc): {accuracy_score(ytest, prds)}")
