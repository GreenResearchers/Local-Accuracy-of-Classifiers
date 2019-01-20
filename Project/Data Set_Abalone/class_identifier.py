import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import  math


headers = ["sex", "length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight", "rings"]

df = pd.read_csv("abalone.data", header=None, names=headers, na_values="?")

replace_lebels = {"sex":     {"M": 0, "F": 1, "I":2}}
df.replace(replace_lebels, inplace=True)
new_df = pd.DataFrame(data=pd.get_dummies(df, columns=["sex"]))

max_value = new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].max()
min_value = new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].min()

std = df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]].std()
#std= 4 * std.values
range= max_value-min_value
new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]]= \
    ((new_df[["length", "diameter", "height", "whole_weight",
           "shucked_weight", "viscera_weight", "shell_weight"]]-min_value)/std)

#hvdm_check_levels= np.array(df.drop(["length", "diameter", "height", "whole_weight",
#           "shucked_weight", "viscera_weight", "shell_weight"], 1))
#total_male= 0
#total_female= 0
#total_infant= 0
#total_male_majority = 0
#total_male_minority= 0
#total_female_majority= 0
#total_female_minority= 0
#total_infant_majority= 0
#total_infant_minority= 0

#for item in hvdm_check_levels:
#    if item[0] == 0:
#        total_male = total_male + 1
#        if item[1] <= 4 or item[1]>= 16:
#            total_male_minority = total_male_minority + 1
#        else:
#            total_male_majority = total_male_majority + 1
#
#    elif item[0] == 1:
#        total_female = total_female + 1
#        if item[1] <= 4 or item[1]>= 16:
#            total_female_minority = total_female_minority + 1
#        else:
#            total_female_majority = total_female_majority + 1

#    elif item[0] == 2:
#        total_infant = total_infant + 1
#        if item[1] <= 4 or item[1]>= 16:
#            total_infant_minority = total_infant_minority + 1
#        else:
#            total_infant_majority = total_infant_majority + 1

#probable_sex_male_majority=total_male_majority/total_male
#probable_sex_male_minority=total_male_minority/total_male
#probable_sex_female_majority = total_female_majority / total_female
#probable_sex_female_minority = total_female_minority / total_female
#probable_sex_infant_majority = total_infant_majority / total_infant
#probable_sex_infant_minority = total_infant_minority / total_infant

#DifferenceMaleFemaleMajority = probable_sex_male_majority - probable_sex_female_majority
##SqrdDifferenceMaleFemaleMajority= DifferenceMaleFemaleMajority * DifferenceMaleFemaleMajority

#DifferenceMaleFemaleMinority = probable_sex_male_minority - probable_sex_female_minority
#SqrdDifferenceMaleFemaleMinority= DifferenceMaleFemaleMinority * DifferenceMaleFemaleMinority

#DifferenceMaleInfantMajority = probable_sex_male_majority - probable_sex_infant_majority
#SqrdDifferenceMaleInfantMajority= DifferenceMaleInfantMajority * DifferenceMaleInfantMajority

#DifferenceMaleInfantMinority = probable_sex_male_minority - probable_sex_infant_minority
#SqrdDifferenceMaleInfantMinority= DifferenceMaleInfantMinority * DifferenceMaleInfantMinority

#DifferenceFemaleInfantMajority = probable_sex_female_majority - probable_sex_infant_majority
#SqrdDifferenceFemaleInfantMajority= DifferenceFemaleInfantMajority * DifferenceFemaleInfantMajority

#DifferenceFemaleInfantMinority = probable_sex_female_minority - probable_sex_infant_minority
#SqrdDifferenceFemaleInfantMinority= DifferenceFemaleInfantMinority * DifferenceFemaleInfantMinority

#SumSqrdDifferenceMaleFemale = SqrdDifferenceMaleFemaleMajority + SqrdDifferenceMaleFemaleMinority
#SumSqrdDifferenceMaleInfant = SqrdDifferenceMaleInfantMajority + SqrdDifferenceMaleInfantMinority
#SumSqrdDifferenceFemaleInfant = SqrdDifferenceFemaleInfantMajority + SqrdDifferenceFemaleInfantMinority
#count =0

#def HVDM(a, b):
#    print("a:", a[0], ",", a[1], ",", a[2], ",", a[3], ",", a[4], ",", a[5], ",", a[6], ",", a[7])
#    print("b:", b[0], ",", b[1], ",", b[2], ",", b[3], ",", b[4], ",", b[5], ",", b[6], ",", b[7])
#    z = (a[1:7] - b[1:7]) / std[0:6]

#    if a[0] == b[0]:
#        sqr_std_dist_col0 = 0

#    elif a[0] == 0 and b[0] == 1:
#        sqr_std_dist_col0 = SumSqrdDifferenceMaleFemale

#    elif a[0] == 1 and b[0] == 0:
#        sqr_std_dist_col0 = SumSqrdDifferenceMaleFemale

 #   elif a[0] == 0 and b[0] == 2:
 #       sqr_std_dist_col0 = SumSqrdDifferenceMaleInfant

 #   elif a[0] == 2 and b[0] == 0:
 #       sqr_std_dist_col0 = SumSqrdDifferenceMaleInfant

  #  elif a[0] == 1 and b[0] == 2:
  #      sqr_std_dist_col0 = SumSqrdDifferenceFemaleInfant

 #   elif a[0] == 2 and b[0] == 1:
  #      sqr_std_dist_col0 = SumSqrdDifferenceFemaleInfant
  #  else:
  #      sqr_std_dist_col0 = 1



 #   return math.sqrt((z * z).sum() + sqr_std_dist_col0)


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