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

minority_index = []

minority_index1 = np.where(y<=4)
minority_index2 = np.where(y >=16)

minority_index1_list1 = minority_index1[0].tolist()
minority_index2_list2 = minority_index2[0].tolist()

minority_index= minority_index1_list1 + minority_index2_list2

safe=0
boarderline=0
rare=0
outlier=0

for index in minority_index:
    knn = NearestNeighbors(n_neighbors=6)
    # knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric = mydist)
    knn.fit(X)
    neighbours = (knn.kneighbors([X[index]], return_distance=False))
    new_neighbours = [item for sub_neighbours in neighbours for item in sub_neighbours]

    # remove self index or extra index from the list
    if index in new_neighbours:
        new_neighbours.remove(index)
    else:
        del new_neighbours[5]
    # separete the minority and majority samples from the samples
    minority_count = 0
    majority_count = 0
    for sample in new_neighbours:
        # print("Data lebel:",y[index])
        if (y[sample] > 4 and y[sample] < 16):
            # print("Minority Incremented")
            majority_count += 1
        else:
            # print("Maority Incremented")
            minority_count += 1
    # print("Minority count:", minority_count, "Majority_count", majority_count)

    # catagorize the samples
    if (minority_count == 5 and majority_count == 0) or (minority_count == 4 and majority_count == 1):
        # Append the safe and predict data to list
        safe = safe + 1

    elif (minority_count == 2 and majority_count == 3) or (minority_count == 3 and majority_count == 2):
        # Append the boarderline and predict data to list
        boarderline = boarderline + 1

    elif minority_count == 1 and majority_count == 4:
        # Append the rare and predict data to list
        rare = rare + 1

    else:
        # Append the outlier and predict data to list
        outlier = outlier + 1
print("Total Minority:",minority_index)
print("Total Safe::",safe)
print("Total Boarderline",boarderline)
print("Total Rare:",rare)
print("Total Outlier:",outlier)