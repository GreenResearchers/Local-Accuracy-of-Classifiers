import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import  math
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

headers = ["sex", "length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight", "rings"]

df = pd.read_csv("abalone.data", header=None, names=headers, na_values="?")

replace_lebels = {"sex":     {"M": 0, "F": 1, "I":2}}
df.replace(replace_lebels, inplace=True)
#new_df = pd.DataFrame(data=pd.get_dummies(df, columns=["sex"]))

#max_value = new_df[["length", "diameter", "height", "whole_weight",
#           "shucked_weight", "viscera_weight", "shell_weight"]].max()
#min_value = new_df[["length", "diameter", "height", "whole_weight",
#           "shucked_weight", "viscera_weight", "shell_weight"]].min()

std = df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].std()

#range= max_value-min_value
#new_df[["length", "diameter", "height", "whole_weight",
#           "shucked_weight", "viscera_weight", "shell_weight"]]= \
#    ((new_df[["length", "diameter", "height", "whole_weight",
#           "shucked_weight", "viscera_weight", "shell_weight"]]-min_value)/std)

hvdm_check_levels= np.array(df.drop(["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"], 1))
total_male= 0
total_female= 0
total_infant= 0
total_male_majority = 0
total_male_minority= 0
total_female_majority= 0
total_female_minority= 0
total_infant_majority= 0
total_infant_minority= 0

for item in hvdm_check_levels:
    if item[0] == 0:
        total_male = total_male + 1
        if item[1] <= 4 or item[1]>= 16:
            total_male_minority = total_male_minority + 1
        else:
            total_male_majority = total_male_majority + 1

    elif item[0] == 1:
        total_female = total_female + 1
        if item[1] <= 4 or item[1]>= 16:
            total_female_minority = total_female_minority + 1
        else:
            total_female_majority = total_female_majority + 1

    elif item[0] == 2:
        total_infant = total_infant + 1
        if item[1] <= 4 or item[1]>= 16:
            total_infant_minority = total_infant_minority + 1
        else:
            total_infant_majority = total_infant_majority + 1

probable_sex_male_majority=total_male_majority/total_male
probable_sex_male_minority=total_male_minority/total_male
probable_sex_female_majority = total_female_majority / total_female
probable_sex_female_minority = total_female_minority / total_female
probable_sex_infant_majority = total_infant_majority / total_infant
probable_sex_infant_minority = total_infant_minority / total_infant


def HVDM(a, b):
    sqr_std_dist_col0 = 0

    if a[0] == b[0]:
        sqr_std_dist_col0 = 0

    elif a[0] == 0 and b[0] == 1:
        majority_class_result = probable_sex_male_majority - probable_sex_female_majority
        majority_class_result = majority_class_result * majority_class_result
        minority_class_result = probable_sex_male_minority - probable_sex_female_minority
        minority_class_result = minority_class_result * minority_class_result
        sqr_std_dist_col0 = majority_class_result + minority_class_result

    elif a[0] == 0 and b[0] == 2:
        majority_class_result = probable_sex_male_majority - probable_sex_infant_majority
        majority_class_result = majority_class_result * majority_class_result
        minority_class_result = probable_sex_male_minority - probable_sex_infant_minority
        minority_class_result = minority_class_result * minority_class_result
        sqr_std_dist_col0 = majority_class_result + minority_class_result

    elif a[0] == 1 and b[0] == 0:
        majority_class_result = probable_sex_female_majority - probable_sex_male_majority
        majority_class_result = majority_class_result * majority_class_result
        minority_class_result = probable_sex_female_minority - probable_sex_male_minority
        minority_class_result = minority_class_result * minority_class_result
        sqr_std_dist_col0 = majority_class_result + minority_class_result

    elif a[0] == 1 and b[0] == 2:
        majority_class_result = probable_sex_female_majority - probable_sex_infant_majority
        majority_class_result = majority_class_result * majority_class_result
        minority_class_result = probable_sex_female_minority - probable_sex_infant_minority
        minority_class_result = minority_class_result * minority_class_result
        sqr_std_dist_col0 = majority_class_result + minority_class_result

    elif a[0] == 2 and b[0] == 0:
        majority_class_result = probable_sex_infant_majority - probable_sex_male_majority
        majority_class_result = majority_class_result * majority_class_result
        minority_class_result = probable_sex_infant_minority - probable_sex_male_minority
        minority_class_result = minority_class_result * minority_class_result
        sqr_std_dist_col0 = majority_class_result + minority_class_result

    elif a[0] == 2 and b[0] == 1:
        majority_class_result = probable_sex_infant_majority - probable_sex_female_majority
        majority_class_result = majority_class_result * majority_class_result
        minority_class_result = probable_sex_infant_minority - probable_sex_female_minority
        minority_class_result = minority_class_result * minority_class_result
        sqr_std_dist_col0 = majority_class_result + minority_class_result

    dist_col1=a[1]-b[1]
    std_dist_col1=dist_col1/(4*std[0])
    sqr_std_dist_col1=std_dist_col1*std_dist_col1

    dist_col2 = a[2] - b[2]
    std_dist_col2=dist_col2/(4*std[1])
    sqr_std_dist_col2=std_dist_col2*std_dist_col2

    dist_col3 = a[3] - b[3]
    std_dist_col3 = dist_col3 / (4*std[2])
    sqr_std_dist_col3 = std_dist_col3 * std_dist_col3

    dist_col4 = a[4] - b[4]
    std_dist_col4 = dist_col4 / (4*std[3])
    sqr_std_dist_col4 = std_dist_col4 * std_dist_col4

    dist_col5 = a[5] - b[5]
    std_dist_col5 = dist_col5 / (4*std[4])
    sqr_std_dist_col5 = std_dist_col5 * std_dist_col5

    dist_col6 = a[6] - b[6]
    std_dist_col6 = dist_col6 / (4*std[5])
    sqr_std_dist_col6 = std_dist_col6 * std_dist_col6

    dist_col7 = a[7] - b[7]
    std_dist_col7 = dist_col7 / (4*std[6])
    sqr_std_dist_col7 = std_dist_col7 * std_dist_col7

    total_distance = sqr_std_dist_col0 + sqr_std_dist_col1 + sqr_std_dist_col2 + sqr_std_dist_col3 + sqr_std_dist_col4\
                        + sqr_std_dist_col5 + sqr_std_dist_col6 + sqr_std_dist_col7

    distance = 0
    if total_distance != 0:
        distance = math.sqrt(total_distance)
    return distance

X = np.array(df.drop(['rings'], 1))
y = np.array(df['rings'])

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
    #knn = NearestNeighbors(n_neighbors=6)
    knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric='pyfunc', metric_params={"func":HVDM})
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
print("Total Minority:",len(minority_index))
print("Total Safe::",safe)
print("Total Boarderline",boarderline)
print("Total Rare:",rare)
print("Total Outlier:",outlier)