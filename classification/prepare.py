# im going to make functions that prepares stuff in here
import pandas as pd
import seaborn as sns
import numpy as np

import pandas_profiling

import env
import acquire

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# def prep_iris(df):
#     train, test = train_test_split(iris, train_size=.8, random_state=123)
#     int_encoder = LabelEncoder()
#     int_encoder.fit(train.species)
#     train.species = int_encoder.transform(train.species)
#     return train.species


# def prep_titanic(df):
    
#     dft.drop(columns=['deck'], inplace=True)
#     dft.fillna(np.nan, inplace=True)
#     train, test = train_test_split(dft, train_size=.8, random_state=123)
    
#     imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

#     imp_mode.fit(train[['embarked']])

#     train['embarked'] = imp_mode.transform(train[['embarked']])

#     test['embarked'] = imp_mode.transform(test[['embarked']])
    
#     int_encoder = LabelEncoder()
#     int_encoder.fit(train.embarked)
#     train.embarked = int_encoder.transform(train.embarked)
    
#     embarked_array = np.array(train.embarked)
#     embarked_array[0:5]
    
#     embarked_array = embarked_array.reshape(len(embarked_array), 1)
    
#     ohe = OneHotEncoder(sparse=False, categories='auto')
    
#     embarked_ohe = ohe.fit_transform(embarked_array)
#     embarked_ohe
    
#     test.embarked = int_encoder.transform(test.embarked)
    
#     embarked_array = np.array(test.embarked).reshape(len(test.embarked), 1)
    
#     embarked_test_ohe = ohe.transform(embarked_array)
    
    
#     train_age_fare = train[['age', 'fare']]
#     test_age_fare = test[['age', 'fare']]
#     scaler, train_age_fare_scaled, test_age_fare_scaled = split_scale.my_minmax_scaler(train_age_fare, test_age_fare)
    
#     return embarked_test_ohe

#Do your work for this in a file named `prepare`.
#1)Use the function defined in `aquire.py` to load the iris data.


from acquire import get_iris_data
from acquire import get_titanic_data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler 

iris_df=get_iris_data()

#1a) Drop the `species_id` and `measurement_id` columns.
def drop_columns(df):
    return df.drop(columns=['species_id','measurement_id'])

#1b) Rename the `species_name` column to just `species`.
def rename_columns(df):
    df['species']=df['species_name']
    return df

#1c)Encode the species name using a sklearn encoder. Research the inverse_transform method
#of the label encoder.How might this be useful.
def encode_columns(df):
    encoder=LabelEncoder()
    encoder.fit(df.species)
    df.species=encoder.transform(df.species)
    return df,encoder

#create a function that accepts the untransformed iris
#data, and returns the data with the transformations above applied.
def prep_iris(df):
    df=df.pipe(drop_columns).pipe(rename_columns).pipe(encode_columns)
    return df 

# Titanic Data
# Use the function you defined in aquire.py to load the titanic data set    
df=get_titanic_data()

# 2a) Handle the missing values in the `embark_town` and `embarked`columns.
def titanic_missing_fill(df):
    df.embark_town.fillna('Other',inplace=True)
    df.embarked.fillna('Unknown',inplace=True)
    return df

# 2b) Remove the deck column.
def titanic_remove_columns(df):
    return df.drop(columns=['deck'])

# 2c) Use a label encoder to transform the `embarked` column
def encode_titanic(df):
    encoder_titanic=LabelEncoder()
    encoder_titanic.fit(titanic_df.embarked)
    titanic_df=encoder_titanic.transform(titanic_df.embarked)
    return titanic_df,encoder_titanic

# 2d) Scale the `age` and `fare` columns using a min/max scaler.
def scale_titanic(df):
    scaled=MinMaxScaler()
    scaled.fit(df[['age','fare']])
    df[['age','fare']]=scaled.transform(df[['age','fare']])
    return df,scaled

#Why might this be beneficial? When might this be beneficial? When might you not
#want to do this?
#Create a function named prep_titanic that accepts the untransformed titanic data,
#and returns the data with the transformations above applied
def prep_titanic(df):
    df=df.pipe(titanic_missing_fill).pipe(titanic_remove_columns).pipe(encode_titanic).pipe(scale_titanic)
    return df