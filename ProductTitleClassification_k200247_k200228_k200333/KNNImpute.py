from sklearn.preprocessing import LabelEncoder  # import necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

######################################################## 1 ########################################################

def preserve_label(train_df):
    # function to preserve unique labels for each category level in the training data
    labels_c1=train_df['category_lvl1'].unique()  # get unique category level 1 labels
    labels_c2=train_df['category_lvl2'].unique()  # get unique category level 2 labels
    labels_c3=train_df['category_lvl3'].unique()  # get unique category level 3 labels
    return labels_c1, labels_c2, labels_c3  # return the unique labels for each category level

######################################################## 3 ########################################################

#The purpose of this function is to prepare categorical data for machine learning algorithms, which generally require numerical data.
def encode_utility(data):
    # function to encode non-null data and replace it in the original data
    encoder = LabelEncoder()  # instantiate a LabelEncoder object
    nonulls = np.array(data.dropna())  # get non-null values from the data and convert to numpy array
    impute_reshape = nonulls.reshape(-1,1)  # reshape the array to have a single column
    impute_ordinal = encoder.fit_transform(impute_reshape)  # fit and transform the data using LabelEncoder
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)  # replace non-null values in the original data with encoded values
    return data

######################################################## 2 ########################################################

def encode(target,train_df):
    # function to encode the specified target columns in the training data
    for columns in target:  # iterate through each target column
        encode_utility(train_df[columns])  # encode non-null values in the target column using encode_utility function

######################################################## 4 ########################################################
#Imputing refers to the process of filling in missing values in a dataset with estimated or imputed values. There are different methods to impute missing values, such as mean imputation, median imputation, and KNN imputation.
#KNNImputer is a method for imputing missing values in a dataset by using the k-nearest neighbors approach. The algorithm finds the k-nearest neighbors of each data point with missing values, and then replaces the missing value with the average value of the k-nearest neighbors. This is a common technique used to fill in missing values in datasets, especially in machine learning applications.
def impute(train_df):
    # function to impute missing values in the category levels using KNNImputer
    imputer = KNNImputer()  # instantiate a KNNImputer object
    df_imputed = np.round(imputer.fit_transform(train_df[['category_lvl1', 'category_lvl2', 'category_lvl3']]))  # impute missing values using KNNImputer and round off the results
    return df_imputed  # return the imputed data as numpy array

######################################################## 5 ########################################################

def clean_csv(df,train_df):
    # function to clean the training data and write the result to a CSV file
    df = pd.DataFrame(df, columns = ['category_lvl1','category_lvl2','category_lvl3'])  # create a new DataFrame with category levels as columns
    df ['Title_desc'] = train_df['titleDescp']  # add title and description data to the new DataFrame
    df.to_csv('train_clean.csv', index=False, header=True)  # write the new DataFrame to a CSV file without index and with column names
    return df  # return the new cleaned DataFrame
