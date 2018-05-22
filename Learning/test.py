import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

#Preparing data
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Finding indexes of y_test


#Finding five neighbourhood
count = 0
while count < len(X):
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(X)
    neighbours = (knn.kneighbors([X[count]], return_distance=False))
    new_neighbours = [item for sub_neighbours in neighbours for item in sub_neighbours]
    if count in new_neighbours:
        new_neighbours.remove(count)
    else:
        del new_neighbours[5]
    print(new_neighbours)
    count += 1

