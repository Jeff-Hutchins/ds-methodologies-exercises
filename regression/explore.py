import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from env import user, host, password
import wrangle
import split_scale


# As a customer analyst, I want to know who has spent the most money with us over their 
# lifetime. I have monthly charges and tenure, so I think I will be able to use those 
# two attributes as features to estimate total_charges. I need to do this within an 
# average of $5.00 per customer.

# Create a file, explore.py, that contains the following functions for exploring your 
# variables (features & target).

get_db_url(user, host, password, database="telco_churn")
telco = wrangle_telco()
telco

# 1. Write a function, plot_variable_pairs(dataframe) that plots all of the pairwise 
# relationships along with the regression line for each pair.

def plot_variable_pairs(dataframe):

sns.PairGrid(telco)

# 2. Write a function, months_to_years(tenure_months, df) that returns your dataframe with 
# a new feature tenure_years, in complete years as a customer.

def months_to_years(dataframe):
   telco['tenure_years'] = telco.tenure/12
   return telco

telco['tenure_years'] = telco.tenure//12
telco

# 3. Write a function, plot_categorical_and_continous_vars(categorical_var, continuous_var, df), 
# that outputs 3 different plots for plotting a categorical variable with a continuous 
# variable, e.g. tenure_years with total_charges. For ideas on effective ways to visualize 
# categorical with continuous: https://datavizcatalogue.com/. You can then look into seaborn 
# and matplotlib documentation for ways to create plots.

train, test = split_my_data(telco, train_ratio=.80, seed=123)
train = train.drop('customer_id', axis=1)
test = test.drop('customer_id', axis=1)

train = train[['tenure_years', 'total_charges']]
test = test[['tenure_years', 'total_charges']]
train
test

scaler = standard_scaler(train, test)
scaler
train_scaled = train_scaled.drop('monthly_charges', axis=1)
test_scaled = test_scaled.drop('monthly_charges', axis=1)
train_scaled

def plot_categorical_and_continous_vars(categorical_var, continuous_var, df):

