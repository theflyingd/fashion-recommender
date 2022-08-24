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

    # filter data to include only purchases close to validation period ("seasonal" model)    
    transactions = transactions.query(query_str).copy()
    
    # count customers and articles that are predictable based on this subset of data    
    dim = {'n':transactions.customer_id.nunique(),
           'm':transactions.article_id.nunique(),
           'n_all':customers.customer_id.nunique(),
           'm_all':articles.article_id.nunique()}

    logger.info(f'Number of customers that can be predicted based on this model: {dim["n"]} ({np.round(dim["n"]/dim["n_all"]*100.0, decimals=2)} %).')
    logger.info(f'Number of predictable articles for these customers: {dim["m"]} ({np.round(dim["m"]/dim["m_all"]*100.0, decimals=2)} %).')
    
    # group interactions (purchases) per customer and article
    int = group_interactions(transactions)
    
    return int, dim, transactions, customers

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

    return Y_csr, sparsity, user_ids, user_id_map, item_id_map_rev

def get_als_model(Y, n_factors=1280, reg=0.01, it=30):
    # instantiate and train model
    model = AlternatingLeastSquares(factors=n_factors, regularization=reg, num_threads=0, iterations=it, use_gpu=False)
    model.fit(2 * Y)    
    return model

def idx_to_ids(item_ids, user_ids, map_reverse):
    # convert matrix indices to item ids
    tmp = pd.DataFrame(item_ids, index=user_ids)
    return tmp.apply(lambda s: ' '.join(s.apply(lambda id: map_reverse[id])), axis=1)


def run_training(n_factors=1280, reg=0.01, it=30):
    logger.info('Loading and filtering data...')
    # define training periods based on the given test weeks and make query strings from them
    test_weeks = [{'start_date':'{}-08-26', 'end_date':'{}-09-01'}, {'start_date':'{}-09-02', 'end_date':'{}-09-08'},
    {'start_date':'{}-09-09', 'end_date':'{}-09-15'}, {'start_date':'{}-09-16', 'end_date':'{}-09-22'}, {'start_date':'{}-09-23', 'end_date':'{}-09-29'}] 
    query_strings = ['', '', '', '', '']
    query_strings_full = ['', '', '', '', '']
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
        query_strings_full[i] = f't_dat < "{start_date}"'

    # prepare, train and save models        
    logger.info(f'Instantiating and training models using {n_factors} factors with regularization strength {reg} and up {it} iterations...')

    # repeat for each test week
    for i, qs in enumerate(query_strings):
        logger.info(f'Working on seasonal model for test week {i+1}...')
        int_seas, dim_seas, _, _ = load_and_filter_data(qs)

        # create utility matrix
        Ys, sparse_seas, user_ids, user_id_map, item_id_map_rev  = get_utility_matrix(int_seas, dim_seas['n'], dim_seas['m'])    
        logger.info(f'Utility matrix of seasonal model for test week {i+1} has dimensions ({dim_seas["n"]}, {dim_seas["m"]}) and a sparsity factor of {np.round(sparse_seas, 2)}.')    
        rs = get_als_model(Ys, n_factors=n_factors, reg=reg, it=it)        

        # predict full batch of users
        logger.info(f'Predicting first batch of users for test week {i+1}...')
        user_idx = [user_id_map[id] for id in user_ids]
        ids, _ = rs.recommend(user_idx, Ys[user_idx], N=12, filter_already_liked_items=False)

        # get interaction data for full model (everything before test week) and train model
        logger.info(f'Working on full model for test week {i+1}...')
        int_full, dim_full, transactions, customers = load_and_filter_data(query_strings_full[i])        
        Ys_full, sparse_full, _, user_id_map_full, item_id_map_rev_full = get_utility_matrix(int_full, dim_full['n'], dim_full['m'])    
        logger.info(f'Utility matrix of full model has dimensions ({dim_full["n"]}, {dim_full["m"]}) and a sparsity factor of {np.round(sparse_full, 2)}.')
        rs_full = get_als_model(Ys_full, n_factors=n_factors, reg=reg, it=it)    

        # find all customers that have not been predicted by the seasonal model due to lack of data, but can be predicted with full model
        logger.info(f'Predicting second batch of users for test week {i+1}...')
        user_ids_diff = transactions.set_index('customer_id').drop(user_ids, axis=0).reset_index().customer_id.unique()
        user_idx_diff = [user_id_map_full[id] for id in user_ids_diff]
        ids_diff, _ = rs_full.recommend(user_idx_diff, Ys_full[user_idx_diff], N=12, filter_already_liked_items=False)
        logger.info(f'Saving prediction results for test week {i+1}...')

        # convert from matrix indices to item ids
        predictions = pd.concat([idx_to_ids(ids, user_ids, item_id_map_rev), idx_to_ids(ids_diff, user_ids_diff, item_id_map_rev_full)], axis=0)

        # make frame containing all available individualized recommendations and join with customer table
        ids_all = np.hstack([user_ids, user_ids_diff])        
        submission = pd.DataFrame({'prediction':predictions}, index=ids_all)
        submission = customers.join(submission, on='customer_id', how='left').set_index('customer_id')

        # now fill empty predictions (cold starts) with baseline and write to file
        baseline_prediction = '0706016001 0706016002 0372860001 0610776002 0759871002 0464297007 0372860002 0610776001 0399223001 0706016003 0720125001 0156231001'
        submission.fillna(baseline_prediction, inplace=True)
        filename = f'prediction_test_week_{i+1}_ALS_{n_factors}_factors_r_{reg}_maxit_{it}.csv'
        submission.loc[:, 'prediction'].to_csv(os.path.join('data/', filename))
        logger.info(f'Done with test week {i+1}...')
    logger.info('Done...')
    
    return None


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_training()