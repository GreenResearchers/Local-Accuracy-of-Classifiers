import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import  math



old_headers = ["class", "large_spot", "spot_distribute", "activity", "evolution","previous","historic",
           "sun_disk","area","area_large-spot","c-class","m_class","x_class"]

df = pd.read_csv("solar.data", header=None, names=old_headers, na_values="?")
replace_lebels = {"class":     {"F": 1, "A":0,"B":0,"C":0,"D":0,"E":0,"H":0}}
df.replace(replace_lebels, inplace=True)
replace_lebels = {"large_spot":     {"A": 0, "H":1,"K":2,"R":3,"S":4,"X":5}}
df.replace(replace_lebels, inplace=True)
replace_lebels = {"spot_distribute":     {"C": 0, "I":1, "O":2, "X":3}}
df.replace(replace_lebels, inplace=True)

headers = [ "large_spot", "spot_distribute", "activity", "evolution","previous","historic",
           "sun_disk","area","area_large-spot","c-class","m_class","x_class","class"]
new_order = [1,2,3,4,5,6,7,8,9,10,11,12,0]
df = df[df.columns[new_order]]

nominal_attribute_no=[0,1,2,3,4,5,6,7,8,9,10,11]
numeric_attribute_no=[]
label_attribute_no=12
label="class"
minority_label=1
majority_label=0
K=[5,7,9,11,13,15,17,19]
#
std = df[[]].std()
std= 4 * std.values
# ##Elements for HVDM calculation
indexed_sum_sqrd_difference=[]
indexed_elements=[]
for index in nominal_attribute_no:
    sum_sqrd_difference=[]
    hvdm_check_levels= np.array(df.iloc[:,[index,label_attribute_no]])
    #hvdm_check_levels=np.array(hvdm_check_levels,1)
    #print(hvdm_check_levels)
    category_count=np.array(df.iloc[:,[index,label_attribute_no]].groupby(headers[index]).count().to_records().view(type=np.matrix))
    #print(category_count)
    outputs= np.array(df.iloc[:,[index,label_attribute_no]].groupby(headers[label_attribute_no]).count().to_records().view(type=np.matrix))
    #print(outputs)
    output_classes=[]
    elements=[]

    for data in category_count[0]:
        elements.append(data[0])
    indexed_elements.append([index,elements])
    #print(elements)
    for data in outputs[0]:
        output_classes.append(data[0])
    #print(output_classes)
    elements_frequencies=[]
    for element in elements:
        for item in output_classes:
            frequencies=[]
            count=0
            for data in hvdm_check_levels:
                if data[0]==element and data[1]==item:
                   count=count+1
            total = 0
            for x in category_count[0]:
                if element==x[0]:
                   total=x[1]
            probability=count/total
            #print(element,item,count,total)
            frequencies=[element,item,probability]
            elements_frequencies.append(frequencies)
    #print(elements_frequencies)
    diffrences=[]
    demo_element_frequencies=elements_frequencies
    for item1 in elements_frequencies:
        for item in demo_element_frequencies:
            if item1[0]!=item[0] and item1[1]==item[1]:
                probability_diffrence=item1[2]-item[2]
                probability_diffrence_sqr=probability_diffrence*probability_diffrence
                sqr_diffrence=[item1[0],item[0],item1[1],probability_diffrence_sqr]
                diffrences.append(sqr_diffrence)
    #print(diffrences)
    sqrd_difference=[]
    for item1 in diffrences:
        sum=0
        for item in diffrences:
            if item1[0]==item[0] and item1[1]==item[1]:
                sum=sum+item[3]
        sum_smaples=[item1[0],item1[1],sum]
        sqrd_difference.append(sum_smaples)
    #print(sqrd_difference)
    for item in sqrd_difference:
        if item not in sum_sqrd_difference:
            sum_sqrd_difference.append(item)
    #print(sum_sqrd_difference)
    indexed_sum_sqrd_difference.append([index,sum_sqrd_difference])
#print(indexed_sum_sqrd_difference)


def HVDM(a, b):
    z=0
    count=0
    sqr_std_dist_col_nominal=0
    for attribute in numeric_attribute_no:
        normalized_diff = (a[attribute] - b[attribute]) / std[count]
        normalized_diff=normalized_diff*normalized_diff
        z=normalized_diff+z
        count=count+1

    for attribute in nominal_attribute_no:
        sqr_std_dist_col=0
        if a[attribute] == b[attribute]:
            sqr_std_dist_col = 0
        else:
            elements=[]
            for element in indexed_elements:
                if element[0]==attribute:
                    elements=element[1]
            sum_sqrd_differences=[]
            for sum_sqrd_difference in indexed_sum_sqrd_difference:
                if sum_sqrd_difference[0]==attribute:
                    sum_sqrd_differences=sum_sqrd_difference[1]
            for element1 in elements:
                for element2 in elements:
                    if(element1!=element2):
                        for diffrence in sum_sqrd_differences:
                            if diffrence[0]==element1 and diffrence[1]==element2:
                                sqr_std_dist_col = diffrence[2]
                            else:
                                sqr_std_dist_col = 1
        sqr_std_dist_col_nominal=sqr_std_dist_col_nominal+sqr_std_dist_col

    return math.sqrt(z + sqr_std_dist_col_nominal)

#
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

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
#             #distances.append(HVDM(a,b))
#             z=0
#             c=0
#             sqr_std_dist_col_nominal=0
#             for attribute in numeric_attribute_no:
#                 normalized_diff = (a[attribute] - b[attribute]) / (std[c])
#                 normalized_diff=normalized_diff*normalized_diff
#                 z=normalized_diff+z
#                 c=c+1
#
#             for attribute in nominal_attribute_no:
#                 sqr_std_dist_col=0
#                 if a[attribute] == b[attribute]:
#                     sqr_std_dist_col = 0
#                 else:
#                     elements=[]
#                     for element in indexed_elements:
#                         if element[0]==attribute:
#                             elements=element[1]
#                     sum_sqrd_differences=[]
#                     for sum_sqrd_difference in indexed_sum_sqrd_difference:
#                         if sum_sqrd_difference[0]==attribute:
#                             sum_sqrd_differences=sum_sqrd_difference[1]
#                     for element1 in elements:
#                         for element2 in elements:
#                             if(element1!=element2):
#                                 for diffrence in sum_sqrd_differences:
#                                     if diffrence[0]==element1 and diffrence[1]==element2:
#                                         sqr_std_dist_col = diffrence[2]
#                                     else:
#                                         sqr_std_dist_col = 1
#                 sqr_std_dist_col_nominal=sqr_std_dist_col_nominal+sqr_std_dist_col
#             distances.append(math.sqrt(z + sqr_std_dist_col_nominal))
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
#
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
        distances.append(HVDM(X[index],b))

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
