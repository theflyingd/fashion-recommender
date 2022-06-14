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

from config import TRACKING_URI, EXPERIMENT_NAME

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

def group_interactions(transactions):
    """Derives interactions (purchases) per customer and article.

    Args:
        transactions (DataFrame) : transactions data

    Returns:
        interactions (DataFrame) : frame containing the interactions per customer and article
    """
    int = transactions.groupby(['customer_id', 'article_id']).t_dat.count().reset_index()
    return int.rename(columns={'t_dat':'interactions'})

def load_and_filter_data(query_str):
    """Loads the data and derives the interactions as well as utility matrix dimensions needed to instantiate and train the models.
    Args:
        query_str (string) : query string for seasonal (or other) filtering

    Returns:
        tuple(6) : Returns interaction frame, number of users and items for both the seasonal and the full model.
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
    n_seas = trans_seas.customer_id.nunique()
    m_seas = trans_seas.article_id.nunique()
    n_full = transactions.customer_id.nunique()
    m_full = transactions.article_id.nunique()
    n_all = customers.customer_id.nunique()
    m_all = articles.article_id.nunique()
    n_cold = n_all - n_full

    logger.info(f'Number of customers that will be predicted based on the seasonal model: {n_seas} ({np.round(n_seas/n_all*100.0, decimals=2)} %).')
    logger.info(f'Number of predictable articles for these customers: {m_seas} ({np.round(m_seas/m_all*100.0, decimals=2)} %).')
    logger.info(f'Number of customers that will be predicted based on the full model: {n_full} ({np.round(n_full/n_all*100.0, decimals=2)} %).')
    logger.info(f'Number of predictable articles for these customers: {m_full} ({np.round(m_full/m_all*100.0, decimals=2)} %).')
    logger.info(f'Number of cold customers: {n_cold} ({np.round(n_cold/n_all*100.0, decimals=2)} %).')
    # group interactions (purchases) per customer and article
    int_seas = group_interactions(trans_seas)
    int_full = group_interactions(transactions)
    
    return int_seas, n_seas, m_seas, int_full, n_full, m_full

def get_utility_matrix(ints, n_users, n_items):
    # create utility matrix Y
    # rows represent items, columns represent users
    # note: users with no transactions and items never sold are not included
    user_ids = ints.customer_id.unique()
    item_ids = ints.article_id.unique()
    
    # build mappings between user/item id and columns/rows of utility matrix
    item_id_map = dict([(item_id, i) for i, item_id in enumerate(item_ids)])
    item_id_map_rev = dict([(i, item_id) for i, item_id in enumerate(item_ids)])
    user_id_map = dict([(user_id, j) for j, user_id in enumerate(user_ids)])
    user_id_map_rev = dict([(j, user_id) for j, user_id in enumerate(user_ids)])

    # add associated column/row indices for each row in the interactions frame
    ints['i'] = ints.article_id.apply(lambda id: item_id_map[id])
    ints['j'] = ints.customer_id.apply(lambda id: user_id_map[id])

    # create sparse matrix
    Y = coo_matrix((df_int.interactions, (df_int['i'], df_int['j'])), shape=(n_items,n_users))

    # compute sparsity ratio
    n_total = Y.shape[0]*Y.shape[1]
    n_int = Y.nnz
    sparsity = n_int/n_total

    return Y.T.tocsr(), sparsity
    
def run_training():
    logger.info('Loading and filtering data...')
    # get interaction data for seasonal and full models
    query_str = '(t_dat >= "2018-08-22" and t_dat < "2018-09-23") or (t_dat >= "2019-08-22" and t_dat < "2019-09-23") or (t_dat >= "2020-08-01" and t_dat < "2020-09-02")'
    int_seas, n_seas, m_seas, int_full, n_full, m_full = load_and_filter_data(query_str)
    # create utility matrices
    Y_seas, sparse_seas = get_utility_matrix(int_seas, n_seas, m_seas)    
    logger.info(f'Utility matrix of seasonal model has dimensions ({n_seas}, {m_seas}) and a sparsity factor of {np.round(sparse_seas, 2)}.')
    Y_full, sparse_full = get_utility_matrix(int_full, n_full, m_full)
    logger.info(f'Utility matrix of full model has dimensions ({n_full}, {m_full}) and a sparsity factor of {np.round(sparse_full, 2)}.')
    logger.info('Done...')




if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_training()