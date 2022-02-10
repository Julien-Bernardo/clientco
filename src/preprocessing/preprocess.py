import pandas as pd
from scipy.stats import zscore
import numpy as np


def set_data_type(df_sample: pd.DataFrame) -> pd.DataFrame:
    df_sample['client_id'] = df_sample['client_id'].astype('O')
    df_sample['product_id'] = df_sample['product_id'].astype('O')
    df_sample['branch_id'] = df_sample['branch_id'].astype('O')
    return df_sample

def compute_delivery_time(df_sample: pd.DataFrame) -> pd.DataFrame:
    df_sample['date_order'] = pd.to_datetime(df_sample['date_order'])
    df_sample['date_invoice'] = pd.to_datetime(df_sample['date_invoice'])
    df_sample['delivery_time'] =  (df_sample['date_invoice'] - df_sample['date_order']).astype('timedelta64[D]')
    return df_sample

def remove_outliers(df_sample: pd.DataFrame) -> pd.DataFrame:
    z_scores = zscore(df_sample[['delivery_time', 'sales_net', 'quantity']])

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 4).all(axis=1)
    df_no_outliers = df_sample[filtered_entries]
    return df_no_outliers

def compute_profitable_orders(df_sample) -> pd.DataFrame:
    df_sample['profitable_orders'] = np.where(df_sample['sales_net']>0, 1, 0)
    df_sample['number_items'] = 1
    return df_sample

def compute_time_last_order(df_no_outliers: pd.DataFrame) -> pd.DataFrame:
    df_churn = df_no_outliers.groupby(['client_id', 'date_order']).agg(sum).reset_index()
    df_churn['time_last_order'] = df_churn.groupby('client_id')['date_order'].diff()/ np.timedelta64(1, 'D')
    df_churn['time_last_order'] = df_churn['time_last_order'].fillna(0)
    return df_churn

def compute_time_last_order_stats(df_churn: pd.DataFrame) -> pd.DataFrame:
    time_last_order_stats = df_churn.groupby(['client_id']).agg({'time_last_order':['mean', 'std']})
    df_churn = df_churn.merge(time_last_order_stats, left_on=['client_id'], right_index=True)
    return df_churn

def compute_churn(df_churn: pd.DataFrame) -> pd.DataFrame:
    df_churn['churn'] = df_churn['time_last_order'] - (df_churn[('time_last_order', 'mean')] + 2*df_churn[('time_last_order', 'std')])
    df_churn['churn'] = [1 if x>=0 else 0 for x in df_churn['churn']]
    return df_churn

def train_test_split(df: pd.DataFrame, split_date: str, DATE='date_order', TARGET='churn') -> pd.DataFrame:
    X_train = df.query(f"{DATE} < @split_date").drop(TARGET, axis=1).reset_index(drop=True)
    y_train = df.query(f"{DATE} < @split_date")[TARGET].reset_index(drop=True)
    X_test = df.query(f"{DATE} >= @split_date").drop(TARGET, axis=1).reset_index(drop=True)
    y_test = df.query(f"{DATE} >= @split_date")[TARGET].reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def main(df_sample: pd.DataFrame) -> pd.DataFrame:
    df_sample = set_data_type(df_sample)
    df_sample = compute_delivery_time(df_sample)
    df_sample = remove_outliers(df_sample)
    df_sample = compute_profitable_orders(df_sample)
    df_sample = compute_time_last_order(df_sample)
    df_sample = compute_time_last_order_stats(df_sample)
    df_sample = compute_churn(df_sample)
    return df_sample