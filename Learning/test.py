import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


knn = NearestNeighbors(n_neighbors=6)
knn.fit(X)
print(knn.kneighbors([[4, 1, 2, 1, 2, '1', 1, 1, 1]], return_distance=False))