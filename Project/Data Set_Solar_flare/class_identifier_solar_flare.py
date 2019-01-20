import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


headers = ["class", "large_spot", "spot_distribute", "activity", "evolution","previous","historic",
           "sun_disk","area","area_large-spot","c-class","m_class","x_class"]

df = pd.read_csv("solar.data", header=None, names=headers, na_values="?")

print(df)

replace_lebels = {"class":     {"F": 1, "A":0,"B":0,"C":0,"D":0,"E":0,"H":0}}
df.replace(replace_lebels, inplace=True)

new_df = pd.DataFrame(data=pd.get_dummies(df, columns=["large_spot", "spot_distribute", "evolution", "previous"]))


X = np.array(new_df.drop(['class'], 1))
y = np.array(new_df['class'])

minority_index = []

minority_index1 = np.where(y==1)

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