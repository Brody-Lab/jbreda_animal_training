"""
Author: Jess Breda
Date: May 31, 2023
Description: Functions for creating a trial-level dataframe
from the DataJoint Sessions.protocol_data blob to be used
in plotting and analysis of animal training performance. 
"""

import datajoint as dj
import numpy as np
import pandas as pd

import behav_viz.utils.dj_utils as dju

dj.blob.use_32bit_dims = True  # necessary for pd.blob read

bdata = dj.create_virtual_module("bdata", "bdata")


def create_trials_df_from_dj(
    animal_ids, date_min="2000-01-01", date_max="2030-01-01", verbose=False
):
    """
    Create a trial-level dataframe from the DataJoint Sessions.protocol_data blob
    to be used in plotting and analysis of animal training performance. Note
    dataframe has additional calculated attributes built from what is stored in
    the blob.

    params
    -----
    animal_ids : list
        list of animal ids to fetch data for
    date_min : str (optional, default = "2000-01-01")
        minimum date to fetch data for
    date_max : str (optional, default = "2030-01-01")
        maximum date to fetch data for
    verbose : bool (optional, default = False)
        whether to print out load number and date range
        for each animal

    returns
    -------
    trials_df : pd.DataFrame
        dataframe with trial-level data for all animals fetched in
        given date range

    """
    # TODO maybe add save out here if read in is
    assert type(animal_ids) == list, "animal ids must be in a list"

    # each animal will have a df across all sessions fetched, will
    # concat across animals and return
    animals_trials_df = []

    for animal_id in animal_ids:

        print("animal_id", animal_id)
        ## Fetch
        subject_session_key = {"ratname": animal_id}
        date_min_key = f"sessiondate >= '{date_min}'"
        date_max_key = f"sessiondate <= '{date_max}'"

        # bug on 2024-07-27 let to incorrect crash cleanup
        # see commit 493b36d in Protocols
        bad_sessions = [964578, 964576, 964552, 964551]
        exclusion_key = f"sessid NOT IN ({', '.join(map(str, bad_sessions))})"

        # TODO filter out empty sessions here to avoid having to do it later on

        protocol_blobs = (
            bdata.Sessions
            & subject_session_key
            & date_min_key
            & date_max_key
            & exclusion_key
        ).fetch("protocol_data", as_dict=True)

        if not len(protocol_blobs):
            print(
                f"no sessions found for {animal_id} between {date_min} and {date_max}"
            )
            continue

        # n session long items are fetched together
        sess_ids, dates, trials, protocols = (
            bdata.Sessions
            & subject_session_key
            & date_min_key
            & date_max_key
            & exclusion_key
        ).fetch("sessid", "sessiondate", "n_done_trials", "protocol")

        ## Format
        animals_trials_df.append(
            create_animals_trials_df(
                animal_id,
                protocol_blobs,
                sess_ids,
                dates,
                trials,
                protocols,
                verbose=verbose,
            )
        )

    trials_df = pd.concat(animals_trials_df)

    return trials_df


def create_animals_trials_df(
    animal_id, protocol_blobs, sess_ids, dates, trials, protocols, verbose=True
):
    """
    Create a trial-level dataframe from the DataJoint Sessions.protocol_data blob
    given additional Session information (ids, dates, trials, protocol name).

    This function is a wraper for the following functions:
        - drop_empty_sessions
        - convert_to_dicts
        - convert_to_dfs
        - append_and_clean_protocol_dfs

    params
    -----
    animal_id : str
        animal id that data was fetched for
    protocol_blobs : list of arrays
        list returned when fetching pd data from bdata
        tables with len = n sessions
    sess_ids : list of ints
        list of session ids for each session in protocol_blobs
    dates : list of datetimes
        list of session dates for each session in protocol_blobs
    trials : list of ints
        list of number of trials for each session in protocol_blobs
    protocols : list of strs
        list of the protocol being actively run
    verbose : bool (optional, default = True)
        whether to print out load number and date range
        for each animal

    returns
    ------
    animals_trials_df : pd.DataFrame
        dataframe with trial-level data for all sessions fetched for
        a given animal
    """
    protocol_blobs, sess_ids, dates, trials = drop_empty_sessions(
        protocol_blobs, sess_ids, dates, trials, verbose=verbose
    )

    protocol_dicts = convert_to_dicts(protocol_blobs)

    protocol_dfs = convert_to_dfs(protocol_dicts, sess_ids)

    append_and_clean_protocol_dfs(
        protocol_dfs, animal_id, sess_ids, dates, trials, protocols
    )

    animals_trials_df = pd.concat(protocol_dfs, ignore_index=True)

    print(
        f"Fetched {len(dates)} dates from Sessions table from {min(dates)} to {max(dates)} for {animal_id}\n"
    )

    return animals_trials_df


def drop_empty_sessions(pd_blobs, sess_ids, dates, trials, verbose=False):
    """
    sessions with 0 or 1 trials break the later code because
    of dimension errors, so they need to be dropped
    """

    trial_filter = (trials != 0) & (trials != 1)

    if verbose:
        print(
            f"dropping {len(pd_blobs) - np.sum(trial_filter)} sessions of {len(pd_blobs)} due to <2 trials"
        )

    pd_blobs = np.array(pd_blobs)  # list -> array needed for bool indexing
    pd_blobs = pd_blobs[trial_filter]

    sess_ids = sess_ids[trial_filter]
    dates = dates[trial_filter]
    trials = trials[trial_filter]

    return pd_blobs, sess_ids, dates, trials


def convert_to_dicts(blobs):
    """
    Function that takes protocol data (pd) blob(s) from bdata
    sessions table query and converts them to python
    dictionaries using Alvaro's blob transformation code

    inputs
    ------
    blobs : list of arrays
        list returned when fetching pd data from bdata
        tables with len = n sessions

    returns
    -------
    dicts : list of dictionaries
        list of pd or peh dictionaries where len = n sessions and
        each session has a dictionary
    """
    # type of blob is indicated in nested, messy array structure
    data_type = list(blobs[0].keys())[0]
    assert data_type == "protocol_data", "unknown key pair"

    dicts = []

    for session_blob in blobs:
        sess_dict = dju.transform_blob(session_blob[data_type])
        dicts.append(sess_dict)

    return dicts


def convert_to_dfs(dicts, sess_ids):
    """
    TODO
    """
    dfs = []

    for session_dict in dicts:
        try:
            dfs.append(dju.blob_dict_to_df(session_dict))
        except:
            print(f"error in fetching df for session {sess_ids}, skipping it")
            pass

    return dfs


def append_and_clean_protocol_dfs(dfs, animal_id, sess_ids, dates, trials, protocols):
    """
    Function that takes a protocol_df for a session and cleans
    it to correct for data types and format per JRBs preferences

    inputs
    ------
    protocol_df : data frame
        protocol_data dictionary thats been converted to df for a single session
    animal_id : str
        animal id for which the session corresponds to
    sess_id : str
        id from bdata corresponding to session
    date : datetime object or str
        date corresponding to
    trials : int
        number of trials in the session
    protocol : str
        protocol name for the session

    modifies
    -------
    protocol_df : data frame
        (1) animal, date, session id, protocol columns added
        (2) sa/sb converted from Hz to kHz
        (3) min time to spoke computed
        (4) certain columns converted to ints & categories [no longer updated Jul 2024]

    """
    for df, sess_id, date, n_done_trials, protocol in zip(
        dfs, sess_ids, dates, trials, protocols
    ):
        # print(f"Working on session {sess_id} for {animal_id} on {date}")
        # sometimes lengths can get one off depending on when the
        # session ends in the PNT cycle, clipping the last row if needed
        if (
            sess_id == 963427
        ):  # For R010, session 963427 is from the homepod and causes issues
            continue

        if len(df) == n_done_trials + 1:
            df.drop(df.tail(1).index, inplace=True)

        if len(df) != n_done_trials:
            print(
                f"session {sess_id} df is outside of length tolerance! trials: {n_done_trials} df: {len(df)}!"
            )
            continue

        df.insert(0, "trial", np.arange(1, n_done_trials + 1, dtype=int))
        df.insert(1, "animal_id", [animal_id] * n_done_trials)
        df.insert(2, "date", [date] * n_done_trials)
        df.insert(3, "sess_id", [sess_id] * n_done_trials)
        df.insert(4, "protocol", [protocol] * n_done_trials)

        # create a unique pair column indicating sa,sb (eg. "12-3")
        df["sound_pair"] = df.apply(
            lambda row: str(row.sa) + ", " + str(row.sb), axis=1
        )
        # determine the minimum spoke time if animal poked l & r on a trial
        df.loc[:, "min_time_to_spoke"] = df[["first_lpoke", "first_rpoke"]].min(
            axis=1, skipna=True
        )

        # any negative cpokes should be nans (matlab code updated Aug 2024)
        # so no longer needed after this
        df["cpoke_dur"] = df["cpoke_dur"].astype("Float64")

        # df.loc[df["cpoke_dur"] < 0, "cpoke_dur"] = pd.NA

        # had a typo in HistorySection send summary, fixed on 2024-08-06
        if "delay_grwoth" in df.columns:
            # Rename 'delay_grwoth' to 'delay_growth'
            df.rename(columns={"delay_grwoth": "delay_growth"}, inplace=True)

        # !Note have stopped updating this code after DMS2 due to the read
        # !in and plots having no issues. If a plot needs a different dtype,
        # !then I take care of it within the plotting function. This allows
        # !the code to be more flexible for multiple protocols.

        # convert data types (matlab makes everything a float) and utilize
        # pyarrow backend
        string_columns = ["animal_id", "first_spoke", "go_type"]
        df[string_columns] = df[string_columns].astype("string[pyarrow]")

        int_columns = ["trial", "sess_id", "hits", "violations", "temperror"]
        df[int_columns] = df[int_columns].astype("uint64[pyarrow]")

        df["result"] = df["result"].astype("int")

        bool_columns = [
            "stimuli_on",
            "give_use",
            "give_water_not_drunk",
            "crash_hist",
        ]
        df[bool_columns] = df[bool_columns].astype("bool[pyarrow]")

        category_columns = [
            "sess_id",
            "sound_pair",
            "SMA_set",
            "go_type",
        ]
        df[category_columns] = df[category_columns].astype("category")


def drop_dates(df, date_drop_dict):

    for animal_id, date in date_drop_dict.items():
        print("dropping", animal_id, date)
        df = df[~((df["animal_id"] == animal_id) & (df["date"] == date))]

    return df
