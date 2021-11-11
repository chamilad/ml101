from sklearn import tree

# classify fruit images

# input to the classifier
# 0 - bumpy texture
# 1 - smooth texture
# all ints
features = [
    [140, 1],
    [130, 1],
    [150, 0],
    [170, 0],
    ]

# outputs we want
# 0 - apple
# 1 - orange
labels = [0, 0, 1, 1]


# classifier
clf = tree.DecisionTreeClassifier()

# set learning algorithm
# fit = find patterns in data
clf = clf.fit(features, labels)

# make prediction
infruit = [145, 0]
print(f"the prediction for {infruit} is: {clf.predict([infruit])}")
