import pandas as pd
import numpy as np

def prep_store_data(df):
    # parse the date column and set it as the index
    fmt = '%a, %d %b %Y %H:%M:%S %Z'
    df.sale_date = pd.to_datetime(df.sale_date, format=fmt)
    df = df.sort_values(by='sale_date').set_index('sale_date')

    # add some time components as features
    df['month'] = df.index.strftime('%m-%b')
    df['weekday'] = df.index.strftime('%w-%a')

    # derive the total sales
    df['sales_total'] = df.sale_amount * df.item_price
    
    return df

def get_sales_by_day(df):
    sales_by_day = df.resample('D')[['sales_total']].sum()
    sales_by_day['diff_with_last_day'] = sales_by_day.sales_total.diff()
    return sales_by_day

def get_fitbit_data(df):
    df1 = pd.read_csv('2018-04-26_and_2018-05-26.csv', nrows=31)
    df2 = pd.read_csv('2018-05-27_and_2018-06-26.csv', nrows=31)
    df3 = pd.read_csv('2018-06-27_and_2018-07-27.csv', nrows=31)
    df4 = pd.read_csv('2018-07-28_and_2018-08-26.csv', nrows=30)
    df5 = pd.read_csv('2018-08-27_and_2018-09-26.csv', nrows=31)
    df6 = pd.read_csv('2018-09-27_and_2018-10-27.csv', nrows=31)
    df7 = pd.read_csv('2018-10-28_and-2018-11-27.csv', nrows=31)
    df8 = pd.read_csv('2018-11-28_and_2018-12-28.csv', nrows=9)
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], join="inner", ignore_index=True)
    return df