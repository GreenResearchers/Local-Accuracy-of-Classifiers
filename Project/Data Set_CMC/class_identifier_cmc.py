import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


headers = ["wife_age", "wife_edu", "husband_edu", "no_child", "wife_reli","wife_work","husband_occu",
           "living_index","media_exxpo","class"]

df = pd.read_csv("cmc.data", header=None, names=headers, na_values="?")


replace_lebels = {"class":     {1: 0, 3: 0, 2:4}}
df.replace(replace_lebels, inplace=True)

new_df = pd.DataFrame(data=pd.get_dummies(df, columns=["wife_edu", "husband_edu", "husband_occu", "living_index"]))

max_value = new_df[["wife_age", "no_child"]].max()
min_value = new_df[["wife_age", "no_child"]].min()

std = new_df[["wife_age", "no_child"]].std()
range= max_value-min_value
print(max_value,min_value,std,range)

new_df[["wife_age", "no_child"]]= ((new_df[["wife_age", "no_child"]]-min_value)//std)

X = np.array(new_df.drop(['class'], 1))
y = np.array(new_df['class'])

minority_index = []

minority_index1 = np.where(y==4)

minority_index1_list1 = minority_index1[0].tolist()

minority_index= minority_index1_list1

safe=0
boarderline=0
rare=0
outlier=0

for index in minority_index:
    #knn = NearestNeighbors(n_neighbors=6)
    #knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='pyfunc', metric_params={"func":HVDM})
    knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='euclidean')
    knn.fit(X)
    neighbours = (knn.kneighbors([X[index]], return_distance=False))
    new_neighbours = [item for sub_neighbours in neighbours for item in sub_neighbours]

    # remove self index or extra index from the list
    if index in new_neighbours:
        new_neighbours.remove(index)
    else:
        del new_neighbours[5]
    # separete the minority and majority samples from the samples
    print(new_neighbours)
    minority_count = 0
    majority_count = 0
    for sample in new_neighbours:
        # print("Data lebel:",y[index])
        if (y[sample] == 0):
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
print("Total Minority:",len(minority_index))
print("Total Safe::",safe)
print("Total Boarderline",boarderline)
print("Total Rare:",rare)
print("Total Outlier:",outlier)