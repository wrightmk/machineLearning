import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

# k is the feature (coordinates)
# r is the class
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5], [7,7], [8,6]]}
new_features = [5,4]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2) right algorithm but a very slow one and only 2 dimensions hard coded

            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    #most common group [0], how many there were [0][0]

    return vote_result

results = k_nearest_neighbors(dataset, new_features, k=3)
print(results)

# verbose
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=results)
plt.show()


# resulting in new feature being closest to the r dataset as predicted
