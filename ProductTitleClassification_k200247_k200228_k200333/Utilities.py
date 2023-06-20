import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score    
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss                  
from sklearn.metrics import classification_report

################################################# 2: Feature Selection #################################################
def featureSelection(df):
    # Remove unnecessary columns
    df.drop(['country','sku_id','price','type'], inplace=True, axis=1) # axis=1 for column wise drop

    # Combine 'title' and 'description' into a single column
    df['titleDescp'] = df['title'] + " " + df['description']

    # Drop the original 'title' and 'description' columns
    df.drop(['title', 'description'], inplace=True, axis=1) 

    # Extract the target variables from the DataFrame
    Y1 = df['category_lvl1']
    Y2 = df['category_lvl2']
    Y3 = df['category_lvl3']

    return df, Y1, Y2, Y3


################################################# 3: Pre-processing #################################################
def PreProcessing(content):
    # Create a stemmer object to reduce words to their root form
    ps = PorterStemmer()

    # Regular expression to remove HTML tags and entities from the text
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    # Remove non-alphabetic characters and HTML tags/entities from the text
    stemmed_content = re.sub('[^a-zA-Z]', ' ', str(content))
    stemmed_content = re.sub(CLEANR, '', stemmed_content)

    # Convert text to lowercase and split into individual words
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()

    # Remove stopwords and stem the remaining words
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]

    # Join the stemmed words back into a single string
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content


################################################# 1: Data Cleaning Utility #################################################
def Cleaning_Data_Utility(training_df):
    # Use the featureSelection function to extract the necessary columns from the DataFrame
    X, Y1, Y2, Y3 = featureSelection(training_df) 

    # Apply pre-processing to the 'titleDescp' column
    X['titleDescp'] = X['titleDescp'].apply(PreProcessing)

    # Return the pre-processed text data and the target variables
    return X, Y1, Y2, Y3
