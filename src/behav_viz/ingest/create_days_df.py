"""
Author: Jess Breda
Date: May 31, 2023
Description: Code to create a day-level summary dataframe from
various DataJoint tables to be used in analyzing animal health
and performance over time.
"""

import datajoint as dj
import numpy as np
import pandas as pd
import datetime
import behav_viz.utils.dj_utils as dju
from pathlib import Path
import os


ratinfo = dj.create_virtual_module("intfo", "ratinfo")
bdata = dj.create_virtual_module("bdata", "bdata")

#############
# Functions #
#############


def create_days_df_from_dj(
    animal_ids: list,
    date_min: str = "2000-01-01",
    date_max: str = "2030-01-01",
    verbose: bool = False,
):
    """
    Function to fetch and format day level summary from
    SessionAgg Table in the bdata schema.

    params
    ------
    animal_ids : list (optional, default = None)
        list of animal ids to query.
    date_min : str (optional, default = "2000-01-01")
        minimum date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    date_max : str (optional, default = "2030-01-01")
        maximum date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    verbose : bool (optional, default = False)
        whether to print out verbose statements

    returns
    -------
    daily_summary_df : pd.DataFrame
        data frame containing day-level summary info for all animals
        in `animal_ids` between `date_min` and `date_max`
    """

    # create keys
    SessAgg_df = fetch_session_agg_date_data(
        animal_ids, date_min, date_max, verbose=verbose
    )

    updated_df = create_and_merge_todays_data_if_needed(
        SessAgg_df, animal_ids, date_max, verbose=verbose
    )

    if not len(updated_df) == 0:
        daily_summary_df = update_column_types(updated_df)
        daily_summary_df = rename_columns(daily_summary_df)
        daily_summary_df = compute_additional_columns(daily_summary_df)

        if verbose:
            print(
                f"\n{len(daily_summary_df)} daily summaries fetched for animals: \n{animal_ids}\n"
                f"between {daily_summary_df.date.min().date().strftime('%Y-%m-%d')} and {daily_summary_df.date.max().date().strftime('%Y-%m-%d')}"
            )

        return daily_summary_df
    else:
        print(f"No data found for {animal_ids} between {date_min} and {date_max}")
        return pd.DataFrame()


def fetch_session_agg_date_data(
    animal_ids: list,
    date_min: str = "2000-01-01",
    date_max: str = "2030-01-01",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Function to fetch day level summary data from SessionAggDate table
    in the bdata schema.

    !Note this table is does not contain data from today and is instead updated
    !at the end of each day. If data is needed for today (or a date > max date
    !in table the create_latest_session_agg_date_data() function is used.

    params
    ------
    animal_ids : list (optional, default = None)
        list of animal ids to query.
    date_min : str (optional, default = "2000-01-01")
        minimum date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    date_max : str (optional, default = "2030-01-01")
        maximum date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    verbose : bool (optional, default = False)
        whether to print out verbose statements with fetching stats
    """

    # create keys
    subject_query = [{"ratname": animal_id} for animal_id in animal_ids]
    date_query = f"sessiondate >= '{date_min}' and sessiondate <= '{date_max}'"

    SessAgg_df = pd.DataFrame(
        (bdata.SessionAggDate() & date_query & subject_query).fetch(as_dict=True)
    )

    if len(SessAgg_df) == 0:
        print(
            f"No data found on SessionAggDate for {animal_ids} between {date_min} and {date_max}"
        )
    else:
        if verbose:
            print(
                f"Fetched data from SessionAggDate table from {SessAgg_df.sessiondate.min()} to {SessAgg_df.sessiondate.max()}  "
            )
    return SessAgg_df


def create_and_merge_todays_data_if_needed(
    SessAgg_df, animal_ids: list, date_max: str, verbose=False
):
    """
    Function to aggregate today's data and add it to the SessionAggDate
    table if the queried data includes today. It should be noted that
    the SessionAggDate table is updated at the end of each day and does
    not contain data for the current day.

    params
    ------
    SessAgg_df : pd.DataFrame
        data frame containing the session level summary info directly
        from the SessionAgg table in the bdata schema of DataJoint
    date_max : str
        maximum date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    verbose : bool (optional, default = False)
        whether to print out verbose statements

    returns
    -------
    aggregated_df : pd.DataFrame
        data frame containing the session level summary info from
        the SessionAgg table with today's data appended if it exists
    """

    today = datetime.datetime.now().date()
    max_date_queried = datetime.datetime.strptime(date_max, "%Y-%m-%d").date()

    # Determine if we need to aggregate today's data under which conditions
    if len(SessAgg_df) == 0 and today <= max_date_queried:
        if verbose:
            print(
                f"""
                Database is empty for date range, but user is querying data
                for {today}. Attempting to manually aggregate today's data
                """
            )

        todays_df = dju.aggregate_todays_data(animal_ids)

    elif today > SessAgg_df.sessiondate.max() and today <= max_date_queried:
        if verbose:
            print(
                f"\tLast date on the database is {SessAgg_df.sessiondate.max()} but user is querying data \n\tfor {today}. Attempting to manually aggregate today's data."
            )

        todays_df = dju.aggregate_todays_data(animal_ids)

    else:
        if verbose:
            print("Today is not being queried or data already exists in the database.")
        todays_df = pd.DataFrame()

    # Add today's data to the database if it exists
    if not len(todays_df) == 0:
        if verbose:
            print(f"Today's, {today}, data exits and has been added to SessionAggDate.")
        aggregated_df = pd.concat([SessAgg_df, todays_df], axis=0, ignore_index=True)
        return aggregated_df
    else:
        if verbose:
            print(f"No new data from today, {today} to add to SessionAggDate.")
        return SessAgg_df


def update_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to update the column types of the SessionAgg data frame
    to the appropriate types. This function is necessary because the
    some variables (e.g. Mass) get read in as objects rather than floats
    due to Nans. Additionally, the date columns need to be converted to
    datetime objects.

    params
    ------
    df : pd.DataFrame
        data frame containing the session level summary info directly
        from the SessionAgg table in the bdata schema of DataJoint

    returns
    -------
    df : pd.DataFrame
        data frame with the appropriate column types
    """

    incorrect_object_cols = ["mass", "percent_target", "volume", "totalvol"]
    df[incorrect_object_cols] = df[incorrect_object_cols].astype(float)
    df["sessiondate"] = pd.to_datetime(df["sessiondate"])

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to rename the columns of the SessionAgg data frame to
    more human-readable names.

    params
    ------
    df : pd.DataFrame
        data frame containing the session level summary info directly
        from the SessionAgg table in the bdata schema of DataJoint

    returns
    -------
    df : pd.DataFrame
        data frame with the appropriate column names
    """

    column_name_dict = {
        "ratname": "animal_id",
        "sessiondate": "date",
        "totalvol": "rig_volume",
        "volume": "pub_volume",
        "total_correct": "hit_rate",
        "percent_violations": "viol_rate",
        "hostname": "rigid",
    }

    df = df.rename(columns=column_name_dict)

    return df


def compute_additional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute additional columns for the SessionAgg data frame
    that are not directly available in the table. This includes X
    params
    ------
    df : pd.DataFrame
        data frame containing the session level summary info directly
        from the SessionAgg table in the bdata schema of DataJoint

    returns
    -------
    df : pd.DataFrame
        data frame with the additional columns added
    """

    # training duration
    df["train_dur_hrs"] = (df.endtime - df.starttime).dt.total_seconds() / 3600
    df["trial_rate"] = (df.n_done_trials / df.train_dur_hrs).round(2)
    df["starttime_hrs"] = df.starttime.dt.total_seconds() / 3600
    df["endtime_hrs"] = df.endtime.dt.total_seconds() / 3600
    df["side_bias"] = df.right_correct - df.left_correct
    df["volume_target"] = (df.percent_target / 100 * df.mass).round(2)
    df["water_diff"] = df.pub_volume + df.rig_volume - df.volume_target

    return df
