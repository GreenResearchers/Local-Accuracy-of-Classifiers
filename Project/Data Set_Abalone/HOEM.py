import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import  math

headers = ["sex", "length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight", "rings"]

df = pd.read_csv("abalone.data", header=None, names=headers, na_values="?")
replace_lebels = {"sex":     {"M": 0, "F": 1, "I":2}}
df.replace(replace_lebels, inplace=True)
replace_lebels = {"rings":     {0: 50, 1: 50, 2:50, 3:50, 4:50,23:50, 24:50, 25:50, 26:50, 27:50, 28:50, 29:50,
                                5:100, 6:100, 7:100, 8:100, 9:100, 10:100, 11:100, 12:100, 13:100, 14:100, 15:100,
                                16:50, 17:50, 18:50, 19:50, 20:50, 21:50, 22:50}}
df.replace(replace_lebels, inplace=True)



nominal_attribute_no=[0]
numeric_attribute_no=[1,2,3,4,5,6,7]
label_attribute_no=8
minority_label=50
majority_label=100
K=[5,7,9,11,13,15,17,19]

MIN = df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].min()
MAX = df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].max()
range= MAX- MIN


# # std= 4 * std.values
##Elements for HVDM calculation


def HOEM(a, b):
    count=0
    z=0
    dist_col_nominal=0
    for attribute in numeric_attribute_no:
        normalized_diff = (a[attribute] - b[attribute]) / range[count]
        normalized_diff=normalized_diff*normalized_diff
        z=normalized_diff+z
        count=count+1
    for attribute in nominal_attribute_no:
        if a[attribute] == b[attribute]:
            dist_col = 0
        else:
            dist_col = 1
        dist_col_nominal=dist_col_nominal+dist_col
    return math.sqrt(z + dist_col_nominal)


X = np.array(df.drop(['rings'], 1))
y = np.array(df['rings'])

#
# GM_all=[]
# for k in K:
#     count =0
#     TP=0
#     TN=0
#     FP=0
#     FN=0
#     for a in X:
#         distances=[]
#         for b in X:
#             distances.append(HOEM(a,b))
#
#         sorted_k_distances=np.argsort(distances)[0:k+1]
#         same_index = np.where(sorted_k_distances==count)
#         sorted_k_distances = np.delete(sorted_k_distances,same_index,0)
#
#         minority_count = 0
#         majority_count = 0
#
#         for sample in sorted_k_distances:
#             if (y[sample] ==majority_label):
#                 majority_count += 1
#             else:
#                 minority_count += 1
#         predicted_label=0
#         true_label=y[count]
#         if(majority_count>minority_count):
#             predicted_label = majority_label
#         else:
#             predicted_label = minority_label
#
#         if(true_label==minority_label and predicted_label==true_label):
#             TP=TP+1
#         elif (true_label==minority_label and predicted_label!=true_label):
#             FN=FN+1
#         elif (true_label==majority_label and predicted_label==true_label):
#             TN=TN+1
#         elif (true_label==majority_label and predicted_label!=true_label):
#             FP=FP+1
#         count = count + 1
#
#     TP_rate=TP/(TP+FN)
#     TN_rate=TN/(FP+TN)
#     GM=math.sqrt(TP_rate*TN_rate)
#     data=[k,GM]
#     GM_all.append(data)
#     print("k:",k,"GM:",GM)
# print(GM_all)

minority_index = []

minority_index1 = np.where(y==minority_label)

minority_index1_list1 = minority_index1[0].tolist()

minority_index= minority_index1_list1

safe=0
boarderline=0
rare=0
outlier=0

for index in minority_index:
    distances=[]
    for b in X:
        distances.append(HOEM(X[index],b))

    sorted_k_distances=np.argsort(distances)[0:6]
    same_index = np.where(sorted_k_distances==index)
    sorted_k_distances = np.delete(sorted_k_distances,same_index,0)

    minority_count = 0
    majority_count = 0

    for sample in sorted_k_distances:
        if (y[sample] ==majority_label):
            majority_count += 1
        else:
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
