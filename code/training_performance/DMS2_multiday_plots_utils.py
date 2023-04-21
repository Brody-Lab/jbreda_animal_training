import numpy as np
import pandas as pd
import datetime as dt


def return_date_window(latest_date=None, n_days_back=None):
    """
    Function to create a date window for querying the DataJoint
    SQL database. Pairs nicely with `fetch_latest_training_data`

    params
    ------
    latest_date : str  (optional, default = None)
        latest date to include in the window, defaults to today
        if left empty
    n_days_back : int (optional, default = None)
        number of days back from `latest_date` to include,
        defaults to all days if left empty

    Note: if you are out of range of your table (e.g min date)
    is before the start of training) it's okay.

    returns
    ------
    min_date : str
        minimum date in the specified date window
    max_date : str
        maximum date in the specified date window

    example usage
    ------------
    `return_date_window(latest_date='2022-11-3', n_days_back=8)`
    `return_date_window(latest_date=None, n_days_back=7)`
    """

    if latest_date:
        date_max = pd.to_datetime(latest_date)
    else:
        date_max = pd.Timestamp.today()

    if n_days_back:
        date_min = date_max - pd.Timedelta(days=n_days_back)
        date_min = date_min.strftime("%Y-%m-%d")
    else:
        date_min = n_days_back  # none

    return date_min, date_max.strftime("%Y-%m-%d")
