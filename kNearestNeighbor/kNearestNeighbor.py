# classification: general purpose is to create a model that best divides our data

# clustering (dividing data into groups)

# k nearest neighbor:
# the closest data point(s) in a data set to your given test point

# use odd numbered ks to avoid split votes
# each value can have their own confidence level
# okay for < 1gb data

#downfalls are that it measures distance using eucloiden distances which is a long tedious process that is more costly as the data set grows becasue it has to compare test points to all points in the dataset (can do a workaround using radius and defining all points outside of the radius as outliers but can still be costly)
# support vector machines scale much better

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) #99999 so that program recognizes it as an outlier and doesnt include it.  Lots of data has missing columns therefore if we dropped them we would have lots of holes in our data
df.drop(['id'], 1, inplace=True)

# capital x for features
# y for labels
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# split up data into testing and training data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

#can pickle here but, already so fast
accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
