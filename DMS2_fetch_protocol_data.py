import datajoint as dj
import numpy as np
import pandas as pd

import blob_transformation as bt


ANIMAL_IDS = ["R610", "R611", "R612", "R613", "R614"]

dj.blob.use_32bit_dims = True  # necessary for pd.blob read

bdata = dj.create_virtual_module("bdata", "bdata")


def drop_empty_sessions(pd_blobs, sess_ids, dates, trials, drop_trial_report=False):
    """
    sessions with 0 or 1 trials break the later code because
    of dimension errors, so they need to be dropped
    """

    trial_filter = (trials != 0) & (trials != 1)

    if drop_trial_report:
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
        sess_dict = bt.transform_blob(session_blob[data_type])
        dicts.append(sess_dict)

    return dicts


def convert_to_dfs(dicts, sess_ids):
    # TODO DOC STRINGS?
    dfs = []

    for session_dict in dicts:
        try:
            dfs.append(bt.blob_dict_to_df(session_dict))
        except:
            print(f"error in fetching df for session {sess_ids}, skipping it")
            pass

    return dfs


def clean_protocol_dfs(dfs, animal_id, sess_ids, dates, trials):
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
        date corresponding to session

    modifies
    -------
    protocol_df : data frame
        (1) crashed trials removed
        (2) animal, date, session id columns added
        (3) sa/sb converted from Hz to kHz
        (4) certain columns converted to ints & categories

    """
    for df, sess_id, date, n_done_trials in zip(dfs, sess_ids, dates, trials):
        assert (
            len(df) == n_done_trials
        ), f"session {sess_id} df is not {n_done_trials} long  its {len(df)}!"

        df.insert(0, "trial", np.arange(1, n_done_trials + 1, dtype=int))
        df.insert(1, "animal_id", [animal_id] * n_done_trials)
        df.insert(2, "date", [date] * n_done_trials)
        df.insert(3, "sess_id", [sess_id] * n_done_trials)

        # create a unique pair column indicating sa,sb (eg. "12-3")
        df["sound_pair"] = df.apply(
            lambda row: str(row.sa) + ", " + str(row.sb), axis=1
        )

        # convert data types (matlab makes everything a float)
        int_columns = [
            "hits",
            "violations",
            "temperror",
            "n_lpokes",
            "n_cpokes",
            "n_rpokes",
        ]
        df[int_columns] = df[int_columns].astype("Int64")

        bool_columns = [
            "valid_early_spoke",
            "is_match",
            "stimuli_on",
            "give_use",
            "replay_on",
            "give_water_not_drunk",
            "crash_hist",
        ]
        df[bool_columns] = df[bool_columns].astype("bool")

        category_columns = [
            "sess_id",
            "result",
            "sound_pair",
            "first_spoke",
            "sides",
            "SMA_set",
            "go_type",
            "give_type_set",
            "give_type_imp",
        ]
        df[category_columns] = df[category_columns].astype("category")
