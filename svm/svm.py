#SVM intro: pretend you have a bunch of points on a 2d graph, +'s and -'s. Draw an iamginary line between the datasets in whichever way you see fit as a seperator and then any new unknown point introduced becomes apart of whichever side of the SVM line it appears on.

# downside: optimization problem, all of your featue sets have to be in memory to optimize.  can be a problem is your dataset gets into the tetrabytes

# ^can use mini batches as a way of tackling this problem.  sequential minimal optimization (SMO)

# -----------
# need to test for all the edge cases i.e. [1,1] => [1,-1], [-1,1], [-1,-1] but can use convex optimization to not necessarily test all the plots but more of a range of where they are.
# -------------
# most importantly the value of our support vectors in the formula U dot W + b = 0 need not be a scalar.  it can just as easily be a sign as anything above 0 is + and below zero -.  where = 0 is on the discriminant?



import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('../kNearestNeighbor/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# capital x for features
# y for labels
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

#can pickle here but, already so fast
accuracy = clf.score(X_test, y_test)

print(accuracy)

# example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures),-1)
#
# prediction = clf.predict(example_measures)
# print(prediction)
