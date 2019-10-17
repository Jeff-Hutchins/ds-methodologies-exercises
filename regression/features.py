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
wrangle.wrangle_telco()
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

def select_kbest_freg_unscaled(X_train, y_train, k):
    '''
    Takes unscaled data (X_train, y_train) and number of features to select (k) as input
    and returns a list of the top k features
    '''
    f_selector = SelectKBest(f_regression, k).fit(X_train, y_train).get_support()
    f_feature = X_train.loc[:,f_selector].columns.tolist()
    return f_feature

from sklearn.feature_selection import SelectKBest, f_regression

f_selector = SelectKBest(f_regression, k=2)

f_selector.fit(X_train, y_train)

f_support = f_selector.get_support()
f_feature = X_train.loc[:,f_support].columns.tolist()

print(str(len(f_feature)), 'selected features')
print(f_feature)

# 2. Write a function, select_kbest_freg()_scaled that takes X_train, y_train (scaled) and k as input 
# and returns a list of the top k features.

def select_kbest_freg_scaled(X_train, y_train, k):
    '''
    Takes unscaled data (X_train, y_train) and number of features to select (k) as input
    and returns a list of the top k features
    '''
    f_selector = SelectKBest(f_regression, k).fit(X_train, y_train).get_support()
    f_feature = X_train.loc[:,f_selector].columns.tolist()
    return f_feature

scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])

f_selector = SelectKBest(f_regression, k=2)

f_selector.fit(X_scaled, y_train)

f_support = f_selector.get_support()
f_feature = X_train.loc[:,f_support].columns.tolist()

print(str(len(f_feature)), 'selected features')
print(f_feature)

### Better function that allows scaled or unscaled

def select_kbest_freg(X, y, k):
    '''
    dataframe of features (X),  dataframe of the target (y), and number of features to select (k) as input
    and returns a list of the top k features
    '''
    f_selector = SelectKBest(f_regression, k).fit(X, y).get_support()
    f_feature = X.loc[:,f_selector].columns.tolist()
    return f_feature

# 3. Write a function, ols_backware_elimination() that takes X_train and y_train (scaled) as 
# input and returns selected features based on the ols backwards elimination method.

def ols_backward_elimination(X_train, y_train):
    '''
    Takes dataframe of features and dataframe of target variable as input,
    runs OLS, extracts each features p-value, removes the column with the highest p-value
    until there are no features remaining with a p-value > 0.05
    It then returns a list of the names of the selected features
    '''
    cols = list(X_train.columns)

    while (len(cols) > 0):
        # create a new dataframe that we will use to train the model...each time we loop through it will 
        # remove the feature with the highest p-value IF that p-value is greater than 0.05.
        # if there are no p-values > 0.05, then it will only go through the loop one time. 
        X_1 = X_train[cols]
        # fit the Ordinary Least Squares Model
        model = sm.OLS(y_train,X_1).fit()
        # create a series of the pvalues with index as the feature names
        p = pd.Series(model.pvalues)
        # get the max p-value
        pmax = max(p)
        # get the feature that has the max p-value
        feature_with_p_max = p.idxmax()
        # if the max p-value is >0.05, the remove the feature and go back to the start of the loop
        # else break the loop with the column names of all features with a p-value <= 0.05
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break

    selected_features_BE = cols
    return selected_features_BE



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

def lasso_cv_coef(X_train, y_train):
    '''
    Takes dataframe of features and dataframe of target variable as input,
    runs lassoCV and returns the coefficients for each feature
    and plots the features with their weights. 
    '''
    reg = LassoCV().fit(X_train, y_train)
    coef = pd.Series(reg.coef_, index=X_train.columns)
    p = sns.barplot(x=X_train.columns, y=reg.coef_)
    return coef, p

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

def optimal_number_of_features(X_train, y_train, X_test, y_test):
    '''discover the optimal number of features, n, using our scaled x and y dataframes, recursive feature
    elimination and linear regression (to test the performance with each number of features).
    We will use the output of this function (the number of features) as input to the next function
    optimal_features, which will then run recursive feature elimination to find the n best features
    '''
    features_range = range(1,len(X_train_scaled.columns)+1)
    # set "high score" to be the lowest possible score
    high_score = 0
    # variables to store the feature list and number of features
    number_of_features = 0
    score_list = []

    for n in features_range:
        model = LinearRegression()
        rfe = RFE(model,n).fit(X_train, y_train)
        
        # transform of rfe will remove features that are least important to the number of features desired
        train_rfe = rfe.transform(X_train)

        # remove the same features from test
        test_rfe = rfe.transform(X_test)

        # Now fit the model to our new dataset
        model.fit(train_rfe,y_train)

        # get the model's "score" which is going to be R-squared value, or explained variance score
        score = model.score(test_rfe,y_test)

        # append the score to a list of scores so that we can see which one performs the best
        score_list.append(score)

        # if score > high score then update the high score to be the new high score, and update number of 
        # features to be the current number

        if(score>high_score):
            high_score = score
            number_of_features = n
            
    return number_of_features


def optimal_features(X_train, y_train, number_of_features):
    '''Taking the output of optimal_number_of_features, as n, and use that value to 
    run recursive feature elimination to find the n best features'''
    cols = list(X_train.columns)
    model = LinearRegression()
    
    #Initializing RFE model
    rfe = RFE(model, number_of_features)

    #Transforming data using RFE
    train_rfe = rfe.fit_transform(X_train,y_train)
    test_rfe = rfe.transform(X_test)
    
    #Fitting the data to model
    model.fit(train_rfe,y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    
    X_train_new = pd.DataFrame(train_rfe, columns=selected_features_rfe)
    X_test_new = pd.DataFrame(test_rfe, columns=selected_features_rfe)
    
    return selected_features_rfe, X_train_new, X_test_new
