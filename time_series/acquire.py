from os import path

import requests
import pandas as pd

BASE_URL = 'https://python.zach.lol'
API_BASE = BASE_URL + '/api/v1'

def get_store_data_from_api():
    url = API_BASE + '/stores'
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['payload']['stores'])

def get_item_data_from_api():
    url = API_BASE + '/items'
    response = requests.get(url)
    data = response.json()

    stores = data['payload']['items']

    while data['payload']['next_page'] is not None:
        print('Fetching page {} of {}'.format(data['payload']['page'] + 1, data['payload']['max_page']))
        url = BASE_URL + data['payload']['next_page']
        response = requests.get(url)
        data = response.json()
        stores += data['payload']['items']

    return pd.DataFrame(stores)

def get_sale_data_from_api():
    url = API_BASE + '/sales'
    response = requests.get(url)
    data = response.json()

    stores = data['payload']['sales']

    while data['payload']['next_page'] is not None:
        print('Fetching page {} of {}'.format(data['payload']['page'] + 1, data['payload']['max_page']))
        url = BASE_URL + data['payload']['next_page']
        response = requests.get(url)
        data = response.json()
        stores += data['payload']['sales']

    return pd.DataFrame(stores)

def get_store_data(use_cache=True):
    if use_cache and path.exists('stores.csv'):
        return pd.read_csv('stores.csv')
    df = get_store_data_from_api()
    df.to_csv('stores.csv', index=False)
    return df

def get_item_data(use_cache=True):
    if use_cache and path.exists('items.csv'):
        return pd.read_csv('items.csv')
    df = get_item_data_from_api()
    df.to_csv('items.csv', index=False)
    return df

def get_sale_data(use_cache=True):
    if use_cache and path.exists('sales.csv'):
        return pd.read_csv('sales.csv')
    df = get_sale_data_from_api()
    df.to_csv('sales.csv', index=False)
    return df

def get_all_data():
    sales = get_sale_data()
    items = get_item_data()
    stores = get_store_data()

    sales = sales.rename(columns={'item': 'item_id', 'store': 'store_id'})

    return sales.merge(items, on='item_id').merge(stores, on='store_id')

def get_opsd_data(use_cache=True):
    if use_cache and path.exists('opsd.csv'):
        return pd.read_csv('opsd.csv')
    df = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
    df.to_csv('opsd.csv', index=False)
    return df

def get_fitbit_data():
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

