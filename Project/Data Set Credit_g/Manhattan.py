import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import  math



headers = ["A1", "A2", "A3", "A4", "A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16",
           "A17","A18","A19","A20","class"]

df = pd.read_csv("credit.data", header=None, names=headers, na_values="?")


#new_df = pd.DataFrame(data=pd.get_dummies(df, columns=["A1", "A3", "A4","A6","A7","A9","A10",
#                                                       "A12","A14","A15","A17","A19","A20"]))
replace_lebels = {"A1":     {"A11":0,"A12":1,"A13":2,"A14":3}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A3":     {"A30":0,"A31":1,"A32":2,"A33":3,"A34":4}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A4":     {"A40":0,"A41":1,"A410":2,"A42":3,"A43":4,"A44":5,"A45":6,"A46":7,"A48":8,"A49":9}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A6":     {"A61":0,"A62":1,"A63":2,"A64":3,"A65":4}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A7":     {"A71":0,"A72":1,"A73":2,"A74":3,"A75":4}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A9":     {"A91":0,"A92":1,"A93":2,"A94":3}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A10":     {"A101":0,"A102":1,"A103":2}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A12":     {"A121":0,"A122":1,"A123":2,"A124":3}}
df.replace(replace_lebels, inplace=True)

replace_lebels = {"A14":     {"A141":0,"A142":1,"A143":2}}
df.replace(replace_lebels, inplace=True)
replace_lebels = {"A15":     {"A151":0,"A152":1,"A153":2}}
df.replace(replace_lebels, inplace=True)
replace_lebels = {"A17":     {"A171":0,"A172":1,"A173":2,"A174":3}}
df.replace(replace_lebels, inplace=True)
replace_lebels = {"A19":     {"A191":0,"A192":1}}
df.replace(replace_lebels, inplace=True)
replace_lebels = {"A20":     {"A201":0,"A202":1}}
df.replace(replace_lebels, inplace=True)

max_value = df[["A2","A5","A8","A11","A13","A16","A18"]].max()

min_value = df[["A2","A5","A8","A11","A13","A16","A18"]].min()

std = df[["A2","A5","A8","A11","A13","A16","A18"]].std()

range= max_value-min_value
df[["A2","A5","A8","A11","A13","A16","A18"]]= \
    ((df[["A2","A5","A8","A11","A13","A16","A18"]]-min_value)/std)

nominal_attribute_no=[0,2,3,5,6,8,9,11,13,14,16,18,19]
numeric_attribute_no=[1,4,7,10,12,15,17]
label_attribute_no=20
minority_label=2
majority_label=1
K=[5,7,9,11,13,15,17,19]

MIN = df[["A2", "A5", "A8", "A11","A13", "A16", "A18"]].min()
MAX = df[["A2", "A5", "A8", "A11","A13", "A16", "A18"]].max()
range= MAX- MIN


# # std= 4 * std.values
##Elements for HVDM calculation


def Manhattan(a, b):
    count=0
    z=0
    dist_col_nominal=0
    for attribute in numeric_attribute_no:
        normalized_diff = (a[attribute] - b[attribute])
        normalized_diff=math.fabs(normalized_diff)
        z=normalized_diff+z
        count=count+1
    for attribute in nominal_attribute_no:
        if a[attribute] == b[attribute]:
            dist_col = 0
        else:
            dist_col = 1
        dist_col_nominal=dist_col_nominal+dist_col
    return math.sqrt(z + dist_col_nominal)


X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# minority_index = []
#
# minority_index1 = np.where(y==minority_label)
#
# minority_index1_list1 = minority_index1[0].tolist()
#
# minority_index= minority_index1_list1

#print(total_data)
# safe=0
# boarderline=0
# rare=0
# outlier=0
#
# for index in minority_index:
GM_all=[]
for k in K:
    count =0
    TP=0
    TN=0
    FP=0
    FN=0
    for a in X:
        distances=[]
        for b in X:
            distances.append(Manhattan(a,b))

        sorted_k_distances=np.argsort(distances)[0:k+1]
        same_index = np.where(sorted_k_distances==count)
        sorted_k_distances = np.delete(sorted_k_distances,same_index,0)

        minority_count = 0
        majority_count = 0

        for sample in sorted_k_distances:
            if (y[sample] ==majority_label):
                majority_count += 1
            else:
                minority_count += 1
        predicted_label=0
        true_label=y[count]
        if(majority_count>minority_count):
            predicted_label = majority_label
        else:
            predicted_label = minority_label

        if(true_label==minority_label and predicted_label==true_label):
            TP=TP+1
        elif (true_label==minority_label and predicted_label!=true_label):
            FN=FN+1
        elif (true_label==majority_label and predicted_label==true_label):
            TN=TN+1
        elif (true_label==majority_label and predicted_label!=true_label):
            FP=FP+1
        count = count + 1

    TP_rate=TP/(TP+FN)
    TN_rate=TN/(FP+TN)
    GM=math.sqrt(TP_rate*TN_rate)
    data=[k,GM]
    GM_all.append(data)
    print("k:",k,"GM:",GM)
print(GM_all)
#     #knn = NearestNeighbors(n_neighbors=6)
#     knn = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='pyfunc', metric_params={"func":HVDM})
#     #knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='euclidean')
#     knn.fit(X)
#     neighbours = (knn.kneighbors([X[index]], return_distance=False))
#     new_neighbours = [item for sub_neighbours in neighbours for item in sub_neighbours]
#
#     # remove self index or extra index from the list
#     if index in new_neighbours:
#         new_neighbours.remove(index)
#     else:
#         del new_neighbours[5]
#     # separete the minority and majority samples from the samples
#     #print(new_neighbours)
#     minority_count = 0
#     majority_count = 0
#     for sample in new_neighbours:
#         # print("Data lebel:",y[index])
#         if (y[sample] ==majority_label):
#             # print("Minority Incremented")
#             majority_count += 1
#         else:
#             # print("Maority Incremented")
#             minority_count += 1
#     # print("Minority count:", minority_count, "Majority_count", majority_count)
#
#     # catagorize the samples
#     if (minority_count == 5 and majority_count == 0) or (minority_count == 4 and majority_count == 1):
#         # Append the safe and predict data to list
#         safe = safe + 1
#
#     elif (minority_count == 2 and majority_count == 3) or (minority_count == 3 and majority_count == 2):
#         # Append the boarderline and predict data to list
#         boarderline = boarderline + 1
#
#     elif minority_count == 1 and majority_count == 4:
#         # Append the rare and predict data to list
#         rare = rare + 1
#
#     else:
#         # Append the outlier and predict data to list
#         outlier = outlier + 1
# print("Total Minority:",len(minority_index))
# print("Total Safe::",safe)
# print("Total Boarderline",boarderline)
# print("Total Rare:",rare)
# print("Total Outlier:",outlier)
