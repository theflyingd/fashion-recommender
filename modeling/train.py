from logging import getLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle
import warnings
import mlflow
from mlflow.sklearn import save_model  # , log_model
import numpy as np

from feature_engineering import (
    fill_missing_values,
    drop_column,
    transform_altitude,
    altitude_high_meters_mean,
    altitude_mean_log_mean,
    altitude_low_meters_mean,
)

from config import TRACKING_URI, EXPERIMENT_NAME

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

def group_interactions(transactions):
    int = transactions.groupby(['customer_id', 'article_id']).t_dat.count().reset_index()
    return int.rename(columns={'t_dat':'interactions'})

def load_and_filter_data(query_str):
    """Loads the data and derives the interactions as well as utility matrix dimensions needed to instantiate and train the models.
    Args:
        query_str (string) : query string for seasonal (or other) filtering
        
    Returns:
        tuple(6) : Returns interaction series, number of users and items for both the seasonal and the full model.
    """
    # import transactions, customer and article data
    transactions = pd.read_csv('data/transactions_train.csv', parse_dates=[0], dtype={'article_id':'string'})
    articles = pd.read_csv('data/articles.csv', dtype={'article_id':'string'})
    customers = pd.read_csv('data/customers.csv')

    # exclude validation week from training data
    transactions = transactions.query('t_dat < "2020-09-02"')
    # filter data to include only purchases in (for previous years) and right before the validation week to reflect seasonal buying    
    trans_seas = transactions.query(query_str).copy()
    # count customers and articles that are predictable with seasonal and full model respectively
    # the full model is trained on the full transaction table (excluding the validation week)
    n_seasonal = trans_seas.customer_id.nunique()
    m_seasonal = trans_seas.article_id.nunique()
    n_full = transactions.customer_id.nunique()
    m_full = transactions.article_id.nunique()
    n_all = customers.customer_id.nunique()
    m_all = articles.article_id.nunique()
    n_cold = n_all - n_full

    logger.info(f'Number of customers that will be predicted based on the seasonal model: {n_seasonal} ({np.round(n_seasonal/n_all*100.0, decimals=2)} %).')
    logger.info(f'Number of predictable articles for these customers: {m_seasonal} ({np.round(m_seasonal/m_all*100.0, decimals=2)} %).')
    logger.info(f'Number of customers that will be predicted based on the full model: {n_full} ({np.round(n_full/n_all*100.0, decimals=2)} %).')
    logger.info(f'Number of predictable articles for these customers: {m_full} ({np.round(m_full/m_all*100.0, decimals=2)} %).')
    logger.info(f'Number of cold customers: {n_cold} ({np.round(n_cold/n_all*100.0, decimals=2)} %).')
    # group interactions (purchases) per customer and article
    int_seas = group_interactions(trans_seas)
    int_full = group_interactions(transactions)
    
    return int_seas, n_seasonal, m_seasonal, int_full, n_full, m_full


def run_training():
    pass


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    query_str = '(t_dat >= "2018-08-22" and t_dat < "2018-09-23") or (t_dat >= "2019-08-22" and t_dat < "2019-09-23") or (t_dat >= "2020-08-01" and t_dat < "2020-09-02")'
    load_and_filter_data(query_str)