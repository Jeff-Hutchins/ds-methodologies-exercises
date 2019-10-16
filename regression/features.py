from pydataset import data
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from env import user, password, host
import wrangle
import split_scale
from statsmodels.formula.api import ols
from math import sqrt
from sklearn.feature_selection import SelectKBest


# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us over their 
# lifetime. I have monthly charges and tenure, so I think I will be able to use those two 
# attributes as features to estimate total_charges. I need to do this within an average 
# of $5.00 per customer.

get_db_url(user, host, password, database="telco_churn")
telco = wrangle_telco()
telco

telco.head()
telco.describe()
telco.info()
telco.dtypes
telco.columns.values


train, test = train_test_split(telco, train_size=0.80, random_state=123)
train = train.drop('customer_id', axis=1)
test = test.drop('customer_id', axis=1)
train.head()
X_train = train.drop(columns='total_charges')
X_train.head()
y_train = train[['total_charges']]
y_train.head()

# 1. Write a function, select_kbest_freg_unscaled() that takes X_train, y_train and k as input 
# (X_train and y_train should not be scaled!) and returns a list of the top k features.

def select_kbest_freg_unscaled(X_train, y_train):

from sklearn.feature_selection import SelectKBest, f_regression

f_selector = SelectKBest(f_regression, k=2)

f_selector.fit(X_train, y_train)

f_support = f_selector.get_support()
f_feature = X_train.loc[:,f_support].columns.tolist()

print(str(len(f_feature)), 'selected features')
print(f_feature)

# 2. Write a function, select_kbest_freg()_scaled that takes X_train, y_train (scaled) and k as input 
# and returns a list of the top k features.

def select_kbest_freg_scaled():

scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])

f_selector = SelectKBest(f_regression, k=2)

f_selector.fit(X_scaled, y_train)

f_support = f_selector.get_support()
f_feature = X_train.loc[:,f_support].columns.tolist()

print(str(len(f_feature)), 'selected features')
print(f_feature)

# 3. Write a function, ols_backware_elimination() that takes X_train and y_train (scaled) as 
# input and returns selected features based on the ols backwards elimination method.

import statsmodels.api as sm

def ols_backware_elimination():

# create the OLS object:
ols_model = sm.OLS(y_train, X_train)

# fit the model:
fit = ols_model.fit()

# summarize:
fit.summary()

cols = list(X_train.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X_train[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y_train,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols
print(selected_features_BE)

# 4. Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns 
# the coefficients for each feature, along with a plot of the features and their weights.

def lasso_cv_coef():

from sklearn.linear_model import LassoCV

reg = LassoCV()
reg.fit(X_train, y_train)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X_train,y_train))
coef = pd.Series(reg.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (4.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

# 5. Write 3 functions, the first computes the number of optimum features (n) using rfe, the 
# second takes n as input and returns the top n features, and the third takes the list of 
# the top n features as input and returns a new X_train and X_test dataframe with those 
# top features , recursive_feature_elimination() that  the optimum number of 
# features (n) and returns the top n features.


