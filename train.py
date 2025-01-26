#!/usr/bin/env python
# coding: utf-8

####### Heart Disease Prediction #######
# The goal of this project is to use some parameters of patient to determine if they are at risk of heart disease or not.

#### Necessary import
import pickle # to manipulate models
import numpy as np # for matrices and numerical manipulations
import pandas as pd # for dataframes
from sklearn.feature_extraction import DictVectorizer # for One-Hot Encoding
from sklearn.model_selection import train_test_split, KFold # for cross-validation techniques
from sklearn.ensemble import RandomForestClassifier # for random forest classifier

#### Parameters
print("Setting parameters")
# Optimal model's hyperparameters values
n_estimators, max_depth, min_samples_leaf = 40, 10, 1
# Number of splits for Kfold Cross-Validation
n_splits = 5
# model file name
output_file = f'rf_model:{n_estimators}_trees_depth_{max_depth}_min_samples_leaf_{min_samples_leaf}.bin'


#### Read the dataframe
df = pd.read_csv("data/raw_merged_heart_dataset.csv")


#### Data Preparation
### Variables Types Conversion
## From Numerical to Categorical
# Map sex values
df["sex"] = df["sex"].replace({1: "male", 0: "female"})
# Target mapping
df["target"] = df["target"].replace({1: "disease", 0: "normal"})
# Chest Pain values mapping
df["cp"] = df["cp"].replace({4: np.NaN, 3: "asymptomatic", 2: "non_anginal_pain", 1: "atypical_angina", 0: "typical_angina"})
## From Categorical to Numerical
# Convert resting blood pressure
df.trestbps = pd.to_numeric(df.trestbps, errors = 'coerce')
# Convert cholesterol
df.chol = pd.to_numeric(df.chol, errors = 'coerce')
# Convert maximum heart rate
df.thalachh = pd.to_numeric(df.thalachh, errors = 'coerce')

### Features
# list of categorical feature variables
numerical = df.select_dtypes("number").columns.to_list()
# list of categorical feature variables
categorical = df.drop(columns = "target").select_dtypes("object").columns.to_list()


### Categorical variables values
# Fasting blood sugar values mapping
df["fbs"] = df["fbs"].replace({"1": "high_fbs", "0": "low_fbs",
                               "?": np.NaN})
# Resting electrocardiographic results mapping
df["restecg"] = df["restecg"].replace({"2": "left_ventricular_hypertrophy",
                                       "1": "st_t_wave_abnormality",
                                       "0": "normal", "?": np.NaN})
# Exercise-induced angina values mapping
df["exang"] = df["exang"].replace({"1": "yes", "0": "no",
                                   "?": np.NaN})
# Slope values mapping
df["slope"] = df["slope"].replace({"3": np.NaN, "2": "downsloping",
                                   "1": "flat", "0": "upsloping", "?": np.NaN})
# Major vessels mapping
df["ca"] = df["ca"].replace({"4": np.NaN, "3": "three_vessels", "2": "two_vessels",
                             "1": "one_vessel", "0": "no_vessel", "?": np.NaN})
# Thalassemia types values mapping
df["thal"] = df["thal"].replace({"7": np.NaN, "6": np.NaN, "3": "reversible_defect",
                                 "2": "fixed_defect", "1": "normal", "0": np.NaN, "?": np.NaN})


### Handling missing values
# Replacing numerical missing values by median
df.fillna(value = df[["trestbps", "chol", "thalachh"]].median(), inplace = True)
# Replacing categorical missing values by mode
df["fbs"] = df["fbs"].fillna(value = df["fbs"].mode()[0])
df["restecg"] = df["restecg"].fillna(value = df["restecg"].mode()[0])
df["exang"] = df["exang"].fillna(value = df["exang"].mode()[0])

# List of columns with missing values
missing_features = ["cp", "slope", "ca", "thal"]
# For each variale
for first_miss in missing_features:
    # Check another variable
    for second_miss in missing_features:
        # Make sure those variables are different
        if first_miss != second_miss:
            # Get index of observations to drop
            index_to_drop = (df[(df[first_miss].isnull()) 
                             & (df[second_miss].isnull())].index)
            # Drop missing data
            df.drop(index_to_drop, axis = 'index', inplace = True)
            
# Replacing categorical missing values by mode
df["cp"] = df["cp"].fillna(value = df["cp"].mode()[0])
df["slope"] = df["slope"].fillna(value = df["slope"].mode()[0])
df["ca"] = df["ca"].fillna(value = df["ca"].mode()[0])
df["thal"] = df["thal"].fillna(value = df["thal"].mode()[0])


### Target Variable encoding
df["target"] = (df["target"] == "disease").astype(int)

### Data Splitting into Train - Validation - Test
# Splitting into full train and test
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
# Splitting into train and test
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)
# Reset indexes
df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
# Get the target values
y_train = df_train.target.values
y_test = df_test.target.values
y_val = df_val.target.values
# Drop `target` from our data sets
del df_train["target"]
del df_test["target"]
del df_val["target"]


## Model Deployment
# Function for training a random forest classifier
def train(df_train, y_train, n_estimators = 40, max_depth = 10, min_samples_leaf = 1):
    """
    This function takes in a training data set, and its target variable, with hyperparameters
    of a random forest classifier and train the model, to return the encoder
    and the classifier trained.
    ---
    df_train: Training data set
    y_train: Training target variable
    n_estimators: Number of trees for the random forest classifier,
                    default: 40
    max_depth: Maximum depth of trees for the random forest classifier,
                    default: 10
    min_samples_leaf: Minimum sample leaves for trees of the random forest classifier,
                    default: 1
    """
    # Convert training set to list of dictionaries
    train_dicts = df_train[categorical + numerical].to_dict(orient = 'records')
    
    # Initialize One-Hot-Encoder (vectorizer)
    One_Hot_encoder = DictVectorizer(sparse = False)
    # One-Hot-Encoder training and train data encoding
    X_train = One_Hot_encoder.fit_transform(train_dicts)

    # Initialize random forest model
    rf = RandomForestClassifier(n_estimators = n_estimators,
                                max_depth = max_depth,
                                min_samples_leaf = min_samples_leaf,
                                random_state = 42,
                                n_jobs = -1)
    # Model training
    rf.fit(X_train, y_train)

    # return one-hot-encoder and random forest model
    return One_Hot_encoder, rf


# Function to make predictions with a random forest classifier
def predict(df, One_Hot_encoder, rf):
    """
    This function takes in a dataframe, a One-Hot-Encoder (dict vectorizer), and
    a random forest model already trained in ore=der to make predictions.
    ---
    df: dataframe to evaluate the model
    One_Hot_Encoder: dict vectorixer to encode categorical variables in the test dataframe
    rf: random forest classifier already trained
    """
     # Convert data to list of dictionaries
    dicts = df[categorical + numerical].to_dict(orient = 'records')

    # One-Hot-Encoding
    X = One_Hot_encoder.transform(dicts)
    # Make predictions
    y_pred = rf.predict(X)
    
    # return predictions
    return y_pred


### Cross - Validation Training
print(f"Performing KFold Cross-Validation")
# Kfold cross-validation initalization
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)

# Initialize scores
scores = []

# Initialize number of folds
fold = 0

# For each iteration of K-fold split and the pair of indexes generated
for train_idx, val_idx in kfold.split(df_full_train):
    # Select train and validation data
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    # Select target variables
    y_train = df_train.target.values
    y_val = df_val.target.values

    # Train model
    One_Hot_encoder, rf = train(df_train, y_train)
    # Make predictions
    y_pred = predict(df_val, One_Hot_encoder, rf)

    # Get score
    acc = round(100 * (y_pred == y_val).mean(), 2)
    # Store score
    scores.append(acc)
    # print auc
    print(f"Accuracy on fold {fold} is {acc} %.")

    # Increment number of fold
    fold += 1
    
# Print scores' means and standard deviations
print("Validation results:")
print('acc mean = %.2f, acc std = +- %.2f' % (np.mean(scores), np.std(scores)))

### Final Model Training
# Optimal random forest model training
One_Hot_encoder, rf = train(df_full_train[categorical + numerical], df_full_train.target,
                            n_estimators = n_estimators, max_depth = max_depth,
                            min_samples_leaf = min_samples_leaf)
# Make predictions
y_pred = predict(df_test, One_Hot_encoder, rf)
# accuracy score
print('Optimal model accuracy = %.2f.' % (100 * (y_pred == y_test).mean()))


### Save the model
# Open file and write into it
with open(output_file, 'wb') as f_out: 
    # Save model
    print("Storing the model into a file")
    pickle.dump((One_Hot_encoder, rf), f_out)
    
print(f"The model is saved to {output_file}.")


# ---