import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

headers = ["sex", "length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight", "rings"]

df = pd.read_csv("abalone.data", header=None, names=headers, na_values="?")

new_df = pd.DataFrame(data=pd.get_dummies(df, columns=["sex"]))

max_value = new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].max()
min_value = new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].min()


range= max_value-min_value
new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]]= \
    ((new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]]-min_value)/range)

X = np.array(new_df.drop(['rings'], 1))
y = np.array(new_df['rings'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    random_state=36851234)

for train_index, test_index in rskf.split(X, y):
    print("\n\nNew iteration:\n\n")
    #Separete train and test indics for X and y
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #initialize the taret classifier and train it
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    #Store the predicted values
    y_pred = clf.predict(X_test)

    #Calculate global accuracy
    accuracy = accuracy_score(y_test, y_pred)
    #accuracy = clf.score(X_test, y_test)
    accuracy = clf.score(X_test, y_test)

    minority_y_test_index = []

    minority_y_test_index1 = np.where(y_test <=4)
    minority_y_test_index2 = np.where(y_test >=16)

    minority_y_test_index1_list1 = minority_y_test_index1[0].tolist()
    minority_y_test_index2_list2 = minority_y_test_index2[0].tolist()

    minority_y_test_index= minority_y_test_index1_list1 + minority_y_test_index2_list2
    y_pred_minority = []
    y_test_minority = []
    majority_test_index = test_index

    for item in minority_y_test_index:
        y_test_minority.append(y_test[item])
        y_pred_minority.append(y_pred[item])

    majority_test_index=np.delete(majority_test_index,minority_y_test_index)

    print(majority_test_index)
    accuracy_minority = accuracy_score(y_test_minority, y_pred_minority)

    y_pred_majority = []
    y_test_majority = []

    for item in majority_test_index:
        y_test_majority.append(y_test[item])
        y_pred_majority.append(y_pred[item])
    accuracy_majority = accuracy_score(y_test_majority, y_pred_majority)

    print("Total Data:",len(test_index))
    print("Majority data:", len(majority_test_index))
    print("Total Minority Data:",len(minority_y_test_index))
    print("Global accuracy:", accuracy)
    print("Global Minority accuracy:",accuracy_minority)
    #print("Global Majority accuracy:",accuracy_majority)
    minority_index_main_data = []

    for index in minority_y_test_index:
        minority_index_main_data.append([test_index[index], index])

    np.asarray(minority_index_main_data)
    #Variebles to save results about minority data
    y_test_safe = []
    np.asarray(y_test_safe)
    y_pred_safe = []
    np.asarray(y_pred_safe)
    safe=0

    y_test_boarderline = []
    np.asarray(y_test_boarderline)
    y_pred_boarderline = []
    np.asarray(y_pred_boarderline)
    boarderline=0

    y_test_rare = []
    np.asarray(y_test_rare)
    y_pred_rare = []
    np.asarray(y_pred_rare)
    rare=0

    y_test_outlier = []
    np.asarray(y_test_outlier)
    y_pred_outlier = []
    np.asarray(y_pred_outlier)
    outlier=0

    #Finding five neighbourhood of the minority data from test samples
    for index in minority_index_main_data:
        knn = NearestNeighbors(n_neighbors=6)
        #knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric = mydist)
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
            if (y[sample] > 4 and y[sample]<16):
                #print("Minority Incremented")
                majority_count += 1
            else:
                #print("Maority Incremented")
                minority_count += 1
        #print("Minority count:", minority_count, "Majority_count", majority_count)

        #catagorize the samples

        if (minority_count == 5 and majority_count == 0)or(minority_count == 4 and majority_count == 1):
            #Append the safe and predict data to list
            y_test_safe.append(y_test[index[1]])
            y_pred_safe.append(y_pred[index[1]])
            safe=safe+1

        elif(minority_count == 2 and majority_count == 3) or (minority_count == 3 and majority_count == 2):

            #Append the boarderline and predict data to list
            y_test_boarderline.append(y_test[index[1]])
            y_pred_boarderline.append(y_pred[index[1]])
            boarderline=boarderline+1

        elif minority_count == 1 and majority_count == 4:

            #Append the rare and predict data to list
            y_test_rare.append(y_test[index[1]])
            y_pred_rare.append(y_pred[index[1]])
            rare=rare+1

        else:

            #Append the outlier and predict data to list
            y_test_outlier.append(y_test[index[1]])
            y_pred_outlier.append(y_pred[index[1]])
            outlier=outlier+1

    # Find the accuracy for safe samples
    accuracy_safe = accuracy_score(y_test_safe, y_pred_safe)
    print("Safe:", safe)
    print("Accuracy for Safe:",accuracy_safe)

    # Find the accuracy for boarderline samples
    accuracy_boarderline = accuracy_score(y_test_boarderline, y_pred_boarderline)
    print("Boarderline:", boarderline)
    print("Accuracy for Boarderline:",accuracy_boarderline)

    # Find the accuracy for rare samples
    accuracy_rare = accuracy_score(y_test_rare, y_pred_rare)
    print("Rare",rare)
    print("Accuracy for Rare:",accuracy_rare)

    # Find the accuracy for outlier samples
    accuracy_outlier = accuracy_score(y_test_outlier, y_pred_outlier)
    print("Outlier",outlier)
    print("Accuracy for outlier:",accuracy_outlier)
