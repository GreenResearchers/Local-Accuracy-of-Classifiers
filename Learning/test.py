import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import neighbors

#Preparing data
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Test indics will come from stratified cross validation
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    random_state=36851234)

for train_index, test_index in rskf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    #print("X_test index:",test_index)
    y_train, y_test = y[train_index], y[test_index]
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    #count the minoity element
    list = np.bincount(y_test)
    ii = np.nonzero(list)[0]
    list = np.vstack((ii , list[ii])).T

    #sort the array
    list=list[list[:, 1].argsort()]
    print(list)

    #append to new list, with minority indics only

    minority_test_index = np.where(a == value)

    #Finding five neighbourhood
    for index in test_index:
        knn = NearestNeighbors(n_neighbors=6)
        knn.fit(X)
        neighbours = (knn.kneighbors([X[index]], return_distance=False))
        new_neighbours = [item for sub_neighbours in neighbours for item in sub_neighbours]
        if index in new_neighbours:
            new_neighbours.remove(index)
        else:
            del new_neighbours[5]
        print(index, ":", new_neighbours)
