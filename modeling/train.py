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
from scipy.sparse import coo_matrix
import joblib
import os
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm

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

    # exclude validation weeks from training data
    transactions = transactions.query('t_dat < "2020-08-26"')
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
    Y = coo_matrix((ints.interactions, (ints['i'], ints['j'])), shape=(n_items,n_users))
    Y_csr = Y.T.tocsr()

    # compute sparsity ratio
    n_total = Y_csr.shape[0]*Y_csr.shape[1]
    n_int = Y_csr.nnz
    sparsity = n_int/n_total*100.0

    return Y_csr, sparsity

def get_als_model(Y, n_factors=1280, reg=0.01, it=30):
    model = AlternatingLeastSquares(factors=n_factors, regularization=reg, num_threads=0, iterations=it, use_gpu=False)
    model.fit(2 * Y)    
    return model

def run_training(n_factors=1280, reg=0.01, it=30):
    logger.info('Loading and filtering data...')
    # get interaction data for seasonal and full models
    test_weeks = [{'start_date':'{}-08-26', 'end_date':'{}-09-01'}, {'start_date':'{}-09-02', 'end_date':'{}-09-08'},
    {'start_date':'{}-09-09', 'end_date':'{}-09-15'}, {'start_date':'{}-09-16', 'end_date':'{}-09-22'}]
    query_strings = ['', '', '', '']
    for i, week in enumerate(test_weeks):
        # window centered around and including the test week for the past years
        for year in [2018, 2019]:
            start_date = pd.to_datetime(week['start_date'].format(year), yearfirst=True)
            end_date = pd.to_datetime(week['end_date'].format(year), yearfirst=True)
            sd = start_date - pd.Timedelta(12, 'D')
            ed = end_date + pd.Timedelta(12, 'D')
            query_strings[i] += f'(t_dat >= "{sd}" and t_dat <= "{ed}") or '
            # for the current year the test week needs to be excluded and no future data from after the test week can be used
        start_date = pd.to_datetime(week['start_date'].format(2020), yearfirst=True)
        end_date = pd.to_datetime(week['end_date'].format(2020), yearfirst=True)
        sd = start_date - pd.Timedelta(30, 'D')                
        query_strings[i] += f'(t_dat >= "{sd}" and t_dat < "{start_date}")'

    # prepare, train and save models        
    logger.info(f'Instantiating and training models using {n_factors} factors with regularization strength {reg} and up {it} iterations...')
    filename = f'ALS_f{n_factors}_r{reg}_it{it}_'
    for i, qs in enumerate(query_strings):
        logger.info(f'Working on seasonal model for test week {i+1}...')
        int_seas, n_seas, m_seas, int_full, n_full, m_full = load_and_filter_data(qs)
        # create utility matrices
        Ys, sparse_seas = get_utility_matrix(int_seas, n_seas, m_seas)    
        logger.info(f'Utility matrix of seasonal model for test week {i+1} has dimensions ({n_seas}, {m_seas}) and a sparsity factor of {np.round(sparse_seas, 2)}.')    
        #rs = get_als_model(Ys, n_factors=n_factors, reg=reg, it=it)
        #joblib.dump(rs, os.path.join('models/', filename + f'tw{i+1}_seas.sav'))
      
    Ys, sparse_full = get_utility_matrix(int_full, n_full, m_full)    
    logger.info(f'Utility matrix of full model has dimensions ({n_full}, {m_full}) and a sparsity factor of {np.round(sparse_full, 2)}.')
    rs = get_als_model(Ys, n_factors=n_factors, reg=reg, it=it)    
    joblib.dump(rs, os.path.join('models/', filename + 'full.sav'))
    logger.info('Done...')
    
    return None


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_training()