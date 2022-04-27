from logging import getLogger
import pandas as pd
import numpy as np
import warnings
import logging
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

def prepare_data():
    """Loads data and prepares basic ingredients for making the baseline
       and random submissions. 

    Returns:
        tuple consisting of:
        list: top 12 item ids 
        Series: all article ids where the article has been sold at least once
        Series: all customer ids that need prediction
    """
    # find all-time top 12 items
    df_trans = pd.read_csv('data/transactions_train.csv', parse_dates=[0], 
                           dtype={'article_id':'string'})
    df_trans['total_sold'] = df_trans.article_id.apply(lambda id: 1)
    top12_items = df_trans.groupby('article_id').total_sold.sum().sort_values(ascending=False).iloc[0:12]

    # collect article_ids of all items that have sold at least once
    # the random submissions will be picked from this list
    art_ids = df_trans.query('total_sold >= 1').article_id.copy()

    # collect customer_ids of all customers that have to be predicted
    df_cus = pd.read_csv('data/customers.csv')
    cus_ids = df_cus.customer_id.unique()
    return top12_items.index.to_list(), art_ids, cus_ids


def create_baseline():
    """Creates predictions for the baseline model, as well as a random
       submission. None is returned, but baseline_submission.csv and
       random_submission.csv will be created in the data subfolder.

    Returns:
        None: returns None 
    """
    logger.info('Creating baseline and random submission...')
    logger.info('Preparing data...')
    top12_items, art_ids, cus_ids = prepare_data()
    logger.info('Done.')

    logger.info('Creating submission files...')
    # baseline prediction for all customers
    prediction = ' '.join(top12_items)
    predictions = pd.DataFrame({'customer_id':cus_ids, 'prediction':np.repeat(prediction, len(cus_ids))})
    predictions.to_csv('data/baseline_submission.csv', index=False)

    # purely random prediction for all customers
    rng = np.random.default_rng(seed=42)
    predictions = [' '.join(art_ids.iloc[rng.integers(low=0, high=len(art_ids), size=12)].values) for p in tqdm(range(len(cus_ids)))]
    predictions = pd.DataFrame({'customer_id':cus_ids, 'prediction':predictions})
    predictions.to_csv('data/random_submission.csv', index=False)
    logger.info('Done.')
    return None

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    create_baseline()