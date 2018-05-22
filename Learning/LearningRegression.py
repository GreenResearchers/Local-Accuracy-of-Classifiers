import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.cross_validation import train_test_spli
# from sklearn import model_selection
from sklearn import neighbors


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)


y_pred=clf.predict(X_test)
print("X_test")
print(X_test)
print("y_test")
print(y_test)
print("y_pred")
print(y_pred)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_train)
distances, indices = nbrs.kneighbors(X_train)

print(X_test[0])

#print("Stratified Repeated K Fold")
#Repeted K fold cross validation
#rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)

#for train_index, test_index in rkf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
 #   X_train, X_test = X[train_index], X[test_index]
  #  y_train, y_test = y[train_index], y[test_index]
   # clf = neighbors.KNeighborsClassifier()
    #clf.fit(X_train, y_train)
 #   accuracy = clf.score(X_test, y_test)
  #  print(accuracy)

#Stratified Repeted K fold cross validation

#print("Stratified Repeated K Fold")
#rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
 #   random_state=36851234)
#for train_index, test_index in rskf.split(X, y):
 #   #print("TRAIN:", train_index, "TEST:", test_index)
  #  X_train, X_test = X[train_index], X[test_index]
   # y_train, y_test = y[train_index], y[test_index]
   # clf = neighbors.KNeighborsClassifier()
   # clf.fit(X_train, y_train)
   # accuracy = clf.score(X_test, y_test
   # print(accuracy)