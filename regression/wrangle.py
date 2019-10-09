import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Throughout the exercises for Regression in Python lessons, you will use the following 
# example scenario: As a customer analyst, I want to know who has spent the most money 
# with us over their lifetime. I have monthly charges and tenure, so I think I will be 
# able to use those two attributes as features to estimate total_charges. I need to do 
# this within an average of $5.00 per customer.

# The first step will be to acquire and prep the data. Do your work for this 
# exercise in a file named wrangle.py.

# 1. Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn 
# database for all customers with a 2 year contract.

data = pd.read_csv("Telco_Data.csv")
data.head()
data.shape
data.describe()
data.info()

    # rounding to 2 decimal places only dropped 11 customers
data[round(data['Monthly_Tenure']/12, 2) == 0]

round(data['Monthly_Tenure']/12, 2)

data['tenure'] = round(data['Monthly_Tenure']/12, 2)

data = data[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
data

# 2. Walk through the steps above using your new dataframe. You may handle the missing 
# values however you feel is appropriate.

data['monthly_charges'] = data['monthly_charges'].str.replace('$',' ').str.strip().str.replace(',','_').astype(float)
data['total_charges'] = data['total_charges'].astype(str)
data['total_charges'] = data['total_charges'].str.replace('$',' ').str.strip().str.replace(',','_').astype(float)

data



# 3. End with a python file wrangle.py that contains the function, wrangle_telco(), 
# that will acquire the data and return a dataframe cleaned with no missing values.

def wrangle_grades():
    data = pd.read_csv("Telco_Data.csv")
    data['tenure'] = round(data['Monthly_Tenure']/12, 2)
    data = data[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
    
    # grades.drop(columns='student_id', inplace=True)
    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    data = data.dropna().astype('int')
    return data


