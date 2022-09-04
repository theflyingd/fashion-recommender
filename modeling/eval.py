from logging import getLogger
import pandas as pd
import numpy as np
import warnings
import mlflow
import sys, os
from config import TRACKING_URI, EXPERIMENT_NAME
import json

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

class DuplicateCustomerError(Exception):
    """ Exception will be raised when customers are appearing
    several times in the submission file."""
    pass

def prepare_data(sub_file, start_date, end_date):
    """Loads a given submission file and adds a ground_truth column, containing
       the list of item_id that the user actually bought during a given time period. 

    Args:
        sub_file (str, optional): Path and name of submission csv.
        start_date (str, optional): First day of evaluation period as datetime.
        end_date (str, optional): Last day of evaluation period as datetime.

    Returns:
        DataFrame: Dataframe of all customers that have bought something during the validation
        period. Index is the customer_id, prediction column gives the predicted item ids and
        ground_truth gives actual item ids (as whitespace-separated string).
    """

    logger.info("Preparing the data...")

    # load submission
    sub = pd.read_csv(sub_file, index_col=0)
    
    # check for duplicate predictions
    if len(sub.index) > len(sub.index.unique()):
        raise DuplicateCustomerError('Some customers are appearing multiple times in your submission file. Stopping...')

    # get rid of empty and nan predictions
    n_pred = len(sub)
    sub.replace('', np.NaN, inplace=True)
    sub.dropna(axis=0, inplace=True)
    n_after = len(sub)
    if n_pred > n_after:
        logger.warning(f'You have empty predictions and or nan predictions in your submission. These will not be scored. Number of dropped rows: {n_pred-n_after}')
    # load transactions
    df_trans = pd.read_csv('data/transactions_train.csv', parse_dates=[0], 
                           dtype={'article_id':'string'})

    # initially, drop all customers that made no purchase during the test period
    test_data = df_trans.query('t_dat >= @start_date and t_dat <= @end_date').copy()
    scored_customers = test_data.customer_id.unique()

    # reduce to include only customers that were actually predicted
    predicted_customers = sub.index.unique()
    predicted_and_scored = np.intersect1d(np.array(predicted_customers), np.array(scored_customers))
    # utter warning if not all scored customers were predicted
    if len(predicted_and_scored) < len(scored_customers):
        percentage_predicted = len(predicted_and_scored)/len(scored_customers) * 100.0
        logger.warning(f'You have not predicted all customers that bought something during the validation period. You have scored only {percentage_predicted} of all relevant customers.')
    sub = sub.loc[predicted_and_scored]

    # now collect item ids that were actually bought during test
    test_data.loc[:, 'article_id'] = test_data.article_id.apply(lambda i: str(i))
    ground_truth = test_data.groupby('customer_id').article_id.agg(lambda c: " ".join(c))
    # add that data as ground_truth column to the submission df
    sub['ground_truth'] = sub.index.map(mapper=ground_truth)
    logger.info("Done.")
    
    return sub

def average_precision_at_k(predicted, actual, cutoff=12):
    """Computes average precision@k for given k and lists of predicted and actual
       item ids.

    Args:
        predicted list: list of predicted (recommended) item ids
        actual list: list of item ids bought during validation period
        cutoff (int, optional): Cutoff k to compute MAP@k. Defaults to 12.

    Returns:
        float: Average precision at cutoff k.
    """
    n_pred = len(predicted)
    if n_pred > cutoff:
        predicted = predicted[0:cutoff]
        n_pred = cutoff
    n_true = len(actual)
    avg_prec = 0.0
    how_many_of_k = 0
    for k in range(min(n_pred, cutoff)):
        # has the k-th item been correctly predicted?
        is_relevant = int((predicted[k] in actual) and (predicted[k] not in predicted[:k]))
        # track how many of the k items have been correctly predicted in total
        how_many_of_k += is_relevant
        # compute precision at cutoff k (=#relevant_recommendations/#of_recommendations)
        avg_prec += how_many_of_k/(k+1) * is_relevant
    # normalize with number of actually bought items (or cutoff)
    return avg_prec / min(n_true, cutoff)

def mean_average_precision_at_k(sub, cutoff=12):
    """Computes mean average precision at cutoff k given a submission, ground truth and
       cutoff.

    Args:
        sub (DataFrame): DataFrame as returned by prepare_data.
        cutoff (int, optional): Cutoff value k. Defaults to 12.

    Returns:
        float: mean average precision at cutoff k
    """
    # compute mean average precision (at given cutoff)
    mean_avg_prec = 0.0
    # iterate over all test customers and sum up contributions 
    for values in sub.itertuples():
        predicted = values.prediction.split(' ')
        actual = values.ground_truth.split(' ')
        mean_avg_prec += average_precision_at_k(predicted, actual, cutoff)
    # normalize with number of customers
    return mean_avg_prec / len(sub)

def run_evaluation(config_file):
    """Load a given submission and evaluate it using the mean average
       precision at cutoff k. Log metrics with mlflow and saved scores to csv.

    Args:
        config_file (str): Path to json config file.

    Returns:
        None
    """

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with open(config_file) as json_file:
        config = json.load(json_file)

    # iterate through all test weeks where sufficient data for validation is available
    map_at_k = []
    sub_idx = []
    for i, week in enumerate(config["test_weeks"]):
        start_date = pd.to_datetime(week['start_date'], yearfirst=True)
        end_date = pd.to_datetime(week['end_date'], yearfirst=True)
        daterange = start_date.strftime('%Y-%m-%d') + '--' + end_date.strftime('%Y-%m-%d')        
        sub_file = config['file_out'] + '_' + daterange + '.csv'
        cutoff = config['cutoff']

        if end_date <= pd.to_datetime('2020-09-22', yearfirst=True):
            logger.info(f'Computing mean average precision for submission file {sub_file}.')
            logger.info(f'Selected test period: {start_date} -- {end_date}')
            logger.info(f'Selected cutoff k: {cutoff}')
            
            sub_idx.append(sub_file)
            with mlflow.start_run(run_name=config['MLFlow_run_name'] + '_' + daterange): 
                # load data
                sub_data = prepare_data(sub_file, start_date=start_date, end_date=end_date)
                # compute MAP@cutoff
                logger.info('Computing mean average precision...')
                mpk = mean_average_precision_at_k(sub_data, cutoff=cutoff)
                map_at_k.append(mpk)
                logger.info(f'Done. Final Score: {mpk}')
    
                mlflow.log_metric(f'MAP_at_{cutoff}', mpk)
                params = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'k': cutoff
                    }
    
                mlflow.log_params(params)
    map_at_k = pd.DataFrame(data=map_at_k, index=sub_idx, columns=[f'MAP_at_{cutoff}'])
    map_at_k.index.name = 'submission_file'
    score_file = config['file_out'] + '_scores.csv'
    map_at_k.to_csv(score_file)
    return None


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

   # check command line arguments
    if len(sys.argv) == 1:
        logger.warning('No config file provided. Trying default...')
        config_file = 'modeling/config.json'
    else:
        config_file = sys.argv[1]

    config_file = str(config_file)
    if not os.path.exists(config_file):
        logger.warning('Config file does not seem to exist. Exiting...')
        raise(FileNotFoundError())

    run_evaluation(config_file)