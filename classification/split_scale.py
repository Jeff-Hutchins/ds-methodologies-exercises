
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer,RobustScaler,MinMaxScaler
from env import user, host, password

# CREATE THE OBJECT, FIT IT, TRANSFORM IT

# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us over 
# their lifetime. I have monthly charges and tenure, so I think I will be able to 
# use those two attributes as features to estimate total_charges. I need to do this 
# within an average of $5.00 per customer.

# Create split_scale.py that will contain the functions that follow. Each scaler 
# function should create the object, fit and transform both train and test. They 
# should return the scaler, train df scaled, test df scaled. Be sure your indices 
# represent the original indices from train/test, as those represent the indices 
# from the original dataframe. Be sure to set a random state where applicable for 
# reproducibility!

# get_db_url(user, host, password, database="telco_churn")
# telco = wrangle_telco()
# telco

# # split_my_data(X, y, train_pct)

# train, test = train_test_split(telco, train_size=0.80, random_state=123)
# train = train.drop('customer_id', axis=1)
# test = test.drop('customer_id', axis=1)

# # train = train.set_index('customer_id')
# # test = test.set_index('customer_id')
# # train.shape
# # test.shape

# # train.drop('customer_id', axis=1)

# # standard_scaler()

# scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
# train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
# test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
# train_scaled
# test_scaled

# # scale_inverse()

# train = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
# test = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])

# # uniform_scaler()

# scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=seed, copy=True).fit(train)
# train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
# test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])


# # gaussian_scaler()

# scaler = PowerTransformer(method, standardize=False, copy=True).fit(train)
# train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
# test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])


# # min_max_scaler()

# scaler = MinMaxScaler(copy=True, feature_range=minmax_range).fit(train)
# train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
# test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])


# # iqr_robust_scaler()

# scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
# train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
# test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])


import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
import numpy as np

def split_my_data(data, train_ratio=.80, seed=123):
    '''the function will take a dataframe and returns train and test dataframe split 
    where 80% is in train, and 20% in test. '''
    return train_test_split(data, train_size = train_ratio, random_state = seed)

## Types of scalers

def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def my_inv_transform(scaler, train_scaled, test_scaled):
    train = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    test = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return scaler, train, test

def uniform_scaler(train, test, seed=123):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=seed, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def gaussian_scaler(train, test, method='yeo-johnson'):
    scaler = PowerTransformer(method, standardize=False, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def my_minmax_scaler(train, test, minmax_range=(0,1)):
    scaler = MinMaxScaler(copy=True, feature_range=minmax_range).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled