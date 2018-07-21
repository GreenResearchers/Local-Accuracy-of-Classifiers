import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

# Read in the CSV file and convert "?" to NaN
#df = pd.read_csv("imports-85.data", header=None, names=headers, na_values="?")
df = pd.read_csv("imports-85.data", header=None, names=headers, na_values="?")
#df = pd.read_csv("imports-85.data")
df.head()

#df.loc[df['normalized_losses'] == '?', 'normalized_losses'] = -9999
#df.loc[:, 'normalized_losses'].replace('?', '-9999', inplace=True)
#df[['bore','stroke','horsepower','peak_rpm','price']]=df[['bore','stroke','horsepower','peak_rpm','price']].replace("?", 0)



cleanup_nums = {"fuel_type":        {"diesel": 0, "gas": 1},
                "aspiration":       {"std": 0, "turbo": 1 },
                "num_doors":     {"four": 0, "two": 1},
                "engine_location":  {"front": 0, "rear": 1},
                "body_style":       {"hardtop":0, "wagon":1, "sedan":2, "hatchback":3, "convertible":4}
                }

#print(pd.get_dummies(df, columns=["drive_wheels"]).head())
#df[["fuel_type","aspiration","num_doors","engine_location"]] = df[["fuel_type","aspiration","num_doors","engine_location"]].astype('category')
df.replace(cleanup_nums, inplace=True)
df.head()

df=df.drop(['normalized_losses'],1).dropna()


new_df = pd.DataFrame(data=pd.get_dummies(df, columns=["make", "drive_wheels", "engine_type", "num_cylinders",
                                           "fuel_system"]))

columns_type = new_df.dtypes
#print(columns_type)
#print(pd.get_dummies(df, columns=["make", "body_style", "drive_wheels", "engine_type", "num_cylinders","fuel_system"]))

#print(new_df.std(axis=0))

max_value = new_df[[ "symboling","wheel_base", "length", "width", "height", "curb_weight", "engine_size",
      "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]].max()
min_value = new_df[[ "symboling","wheel_base", "length", "width", "height", "curb_weight", "engine_size",
      "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]].min()


range= max_value-min_value
new_df[[ "symboling","wheel_base", "length", "width", "height", "curb_weight", "engine_size",
      "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]]= \
    ((new_df[[ "symboling","wheel_base", "length", "width", "height", "curb_weight", "engine_size",
      "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]]-min_value)/range)

#new_df_norm = (new_df - new_df.mean()) / (new_df.max() - new_df.min())
#print(new_df_norm)

def mydist(x, y):
     return np.sum((x-y)**2)

#Split the data into test lebels and a prediction lebel
X = np.array(new_df.drop(['body_style'], 1))
y = np.array(df['body_style'])


#Split the sets into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Test indics will come from stratified cross validation
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    random_state=36851234)

#Repeate the testing and analyzing
for train_index, test_index in rskf.split(X, y):
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
    print(accuracy)


    #count the minoity element, sort it at decending order and first one will be the minority
    total_levels = np.bincount(y_test)
    ii = np.nonzero(total_levels)[0]
    total_levels = np.vstack((ii , total_levels[ii])).T

    #sort the array
    sorted_levels = total_levels[total_levels[:, 1].argsort()]
    sorted_levels=total_levels[total_levels[:, 1].argsort()]
    #print(sorted_levels)


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
        #knn = NearestNeighbors(n_neighbors=6)
        knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric = mydist)
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


    y_test_index = np.where(y_test == sorted_levels[0][0])

    for data in y_test_index[0]:
        print(test_index[data])

    print(y_test)

