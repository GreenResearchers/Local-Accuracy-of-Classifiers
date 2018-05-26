import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import neighbors
from sklearn.metrics import accuracy_score
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
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    #accuracy = clf.score(X_test, y_test)
    print(accuracy)


    #count the minoity element
    total_levels = np.bincount(y_test)
    ii = np.nonzero(total_levels)[0]
    total_levels = np.vstack((ii , total_levels[ii])).T

    #sort the array
    sorted_levels = total_levels[total_levels[:, 1].argsort()]
    print(sorted_levels)

    #append to new list, with minority indics only1
    #for data in y_test:
    #    if data == sorted_levels[0][0]:
    #Find the indexes of minority data in y_test
    minority_lebel=sorted_levels[0][0]
    minority_y_test_index = np.where(y_test == minority_lebel)
    #separete the minority index from main data using minority test indexes of test data
    minority_index_main_data = []
    # Add minority index of the main data and current index of the test data(to be used for determining correct prediction)
    #in a list of list
    for index in minority_y_test_index[0]:
        minority_index_main_data.append([test_index[index], index])
    np.asarray(minority_index_main_data)
    print(minority_index_main_data)

    y_test_safe = []
    np.asarray(y_test_safe)
    y_pred_safe = []
    np.asarray(y_pred_safe)

    y_test_boarderline = []
    np.asarray(y_test_boarderline)
    y_pred_boarderline = []
    np.asarray(y_pred_boarderline)

    y_test_rare = []
    np.asarray(y_test_rare)
    y_pred_rare = []
    np.asarray(y_pred_rare)

    y_test_outlier = []
    np.asarray(y_test_outlier)
    y_pred_outlier = []
    np.asarray(y_pred_outlier)

    #Finding five neighbourhood of the minority data from test samples
    for index in minority_index_main_data:
        knn = NearestNeighbors(n_neighbors=6)
        knn.fit(X)
        neighbours = (knn.kneighbors([X[index[0]]], return_distance=False))
        new_neighbours = [item for sub_neighbours in neighbours for item in sub_neighbours]

        #remove self index or extra index from the list
        if index[0] in new_neighbours:
            new_neighbours.remove(index[0])
        else:
            del new_neighbours[5]

        #separete the minority and majority samples from the samples
        minority_count = 0
        majority_count = 0
        for sample in new_neighbours:
            #print("Data lebel:",y[index])
            if y[index[0]] == minority_lebel:
                #print("Minority Incremented")
                minority_count += 1
            else:
                #print("Maority Incremented")
                majority_count += 1
        #print("Minority count:", minority_count, "Majority_count", majority_count)

        #catagorize the samples

        if (minority_count == 5 and majority_count == 0)or(minority_count == 4 and majority_count == 1):
            #Append the safe and predict data to list
            y_test_safe.append(y_test[index[1]])
            y_pred_safe.append(y_pred[index[1]])

        elif(minority_count == 2 and majority_count == 3) or (minority_count == 3 and majority_count == 2):

            #Append the boarderline and predict data to list
            y_test_boarderline.append(y_test[index[1]])
            y_pred_boarderline.append(y_pred[index[1]])

        elif minority_count == 1 and majority_count == 4:

            #Append the rare and predict data to list
            y_test_rare.append(y_test[index[1]])
            y_pred_rare.append(y_pred[index[1]])

        else:

            #Append the outlier and predict data to list
            y_test_outlier.append(y_test[index[1]])
            y_pred_outlier.append(y_pred[index[1]])

    # Find the accuracy for safe samples
    accuracy_safe = accuracy_score(y_test_safe, y_pred_safe)
    print("Accuracy for Safe:",accuracy_safe)

    # Find the accuracy for boarderline samples
    accuracy_boarderline = accuracy_score(y_test_boarderline, y_pred_boarderline)
    print("Accuracy for Boarderline:",accuracy_boarderline)

    # Find the accuracy for rare samples
    accuracy_rare = accuracy_score(y_test_rare, y_pred_rare)
    print("Accuracy for Rare:",accuracy_rare)

    # Find the accuracy for outlier samples
    accuracy_outlier = accuracy_score(y_test_outlier, y_pred_outlier)
    print("Accuracy for outlier:",accuracy_outlier)