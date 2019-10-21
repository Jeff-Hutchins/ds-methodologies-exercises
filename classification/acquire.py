import pandas as pd
import numpy as np

import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def aquire_database():
    query = '''
    select *
    from properties_2017
    limit 10;
    '''
    df = pd.read_sql(query, get_db_url('zillow'))
    return df