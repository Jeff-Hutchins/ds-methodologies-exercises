from pydataset import data
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from env import user, host, password
import wrangle
import split_scale
from statsmodels.formula.api import ols
from math import sqrt

# 1. Load the tips dataset from either pydataset or seaborn.

tips = data('tips')
tips = tips[['total_bill', 'tip']]
tips.head()
tips.describe()
tips.info()
tips.dtypes
tips.columns.values

# 2. Fit a linear regression model (ordinary least squares) and compute yhat, predictions of 
# tip using total_bill. You may follow these steps to do that:

    # import the method from statsmodels: from statsmodels.formula.api import ols

from statsmodels.formula.api import ols

    # fit the model to your data, where x = total_bill and y = tip: 
    # regr = ols('y ~ x', data=df).fit()

x = pd.DataFrame(tips.total_bill)
y = pd.DataFrame(tips.tip)
regr = ols('y ~ x', data=tips).fit()

    # compute yhat, the predictions of tip using total_bill: df['yhat'] = regr.predict(df.x)

tips['y'] = y
tips['yhat'] = regr.predict(x)
regr.summary()
tips.head()

# 3. Create a file evaluate.py that contains the following functions.

# 4. Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, 
# and the dataframe as input and returns a residual plot. (hint: seaborn has an easy way 
# to do this!)

def plot_residuals(x, y, dataframe):
    return sns.residplot(x, y, dataframe)

sns.residplot(x, y, tips)

# 5. Write a function, regression_errors(y, yhat), that takes in y and yhat, returns the 
# sum of squared errors (SSE), explained sum of squares (ESS), total sum of squares (TSS), 
# mean squared error (MSE) and root mean squared error (RMSE).

def regression_errors(y, yhat):

tips['residual'] = tips['yhat'] - tips.y
tips['residual-2'] = tips['residual'] ** 2
SSE = sum(tips['residual-2'])
MSE = SSE/len(tips)
RMSE = sqrt(MSE)
print(tips.head())
print(SSE, MSE, RMSE)

# 6. Write a function, baseline_mean_errors(y), that takes in your target, y, computes 
# the SSE, MSE & RMSE when yhat is equal to the mean of all y, and returns the error 
# values (SSE, MSE, and RMSE).

def baseline_mean_errors(y):



# 7. Write a function, better_than_baseline(SSE), that returns true if your model 
# performs better than the baseline, otherwise false.



# 8. Write a function, model_significance(ols_model), that takes the ols model as 
# input and returns the amount of variance explained in your model, and the value 
# telling you whether the correlation between the model and the tip value are 
# statistically significant.


