# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us over their 
# lifetime. I have monthly charges and tenure, so I think I will be able to use those two 
# attributes as features to estimate total_charges. I need to do this within an average 
# of $5.00 per customer.

# 1. Write a function, select_kbest_freg_unscaled() that takes X_train, y_train and k as input 
# (X_train and y_train should not be scaled!) and returns a list of the top k features.

def select_kbest_freg_unscaled():


# 2. Write a function, select_kbest_freg()_scaled that takes X_train, y_train (scaled) and k as input 
# and returns a list of the top k features.

def select_kbest_freg_scaled():


# 3. Write a function, ols_backware_elimination() that takes X_train and y_train (scaled) as 
# input and returns selected features based on the ols backwards elimination method.

def ols_backware_elimination():


# 4. Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns 
# the coefficients for each feature, along with a plot of the features and their weights.

def lasso_cv_coef():


# 5. Write 3 functions, the first computes the number of optimum features (n) using rfe, the 
# second takes n as input and returns the top n features, and the third takes the list of 
# the top n features as input and returns a new X_train and X_test dataframe with those 
# top features , recursive_feature_elimination() that  the optimum number of 
# features (n) and returns the top n features.


