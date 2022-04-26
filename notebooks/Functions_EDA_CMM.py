import pandas as pd
import numpy as np


# Define function for generating dataframe which calculates basketsizes per order.

def calc_basketsize (purchases):
    """Function to generate dataframe with basketsizes out of dataframe with single purchases (e.g. dataframe from transaction_train.csv). 
    Assumption: Purchases of an individual customer on one day form an order.

    Args:
        purchases (_dataframe_): Dataframe which contains single purchases per customer in each row
    """    
    
    purchases['datetime'] = pd.to_datetime(purchases['t_dat'])

    orderbaskets = purchases.groupby(['datetime', 'customer_id']).size().reset_index()
    orderbaskets.rename(columns={0: "basketsize"}, inplace=True)

    return orderbaskets


# Define function for generating dataframe which calculates number of orders per customer.

def calc_orders_cust (purchases):
    """Function to generate dataframe with number of orders out of dataframe with single purchases. Assumption: Purchases of an individual customer on one day form an order.

    Args:
        purchases (_dataframe_): Dataframe which contains single purchases per customer in each row
    """    

    purchases['datetime'] = pd.to_datetime(purchases['t_dat'])

    orderbaskets = purchases.groupby(['datetime', 'customer_id']).size().reset_index()
    orderbaskets.rename(columns={0: "basketsize"}, inplace=True)
    number_orders = orderbaskets.customer_id.value_counts().reset_index()
    number_orders.rename(columns={"index": "customer_id", "customer_id": "number_orders"}, inplace=True)


    return number_orders