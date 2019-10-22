# get_titanic_data: returns the titanic data from the codeup 
# data science database as a pandas data frame.

import pandas as pd
import numpy as np
import seaborn as sns
import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_titanic_data():
    query = '''
    select *
    from passengers;
    '''
    df = pd.read_sql(query, get_db_url('titanic_db'))
    return df

# get_iris_data: returns the data from the iris_db on the codeup 
# data science database as a pandas data frame. The returned 
# data frame should include the actual name of the species in 
# addition to the species_ids.

def get_iris_data():
    df = sns.load_dataset("iris")
    return df

