import datajoint as dj
import numpy as np
import pandas as pd

import dj_utils as djut
from dj_utils import ANIMAL_IDS

dj.blob.use_32bit_dims = True  # necessary for pd.blob read

bdata = dj.create_virtual_module("bdata", "bdata")


def fetch_latest_trials_data(
    animal_ids=None, date_min="2000-01-01", date_max="2030-01-01", verbose=False
):
    """
    TODO
    """
    # TODO maybe add save out here if read in is

    animal_ids = ANIMAL_IDS if animal_ids is None else animal_ids
    assert type(animal_ids) == list, "animal ids must be in a list"

    # each animal will have a df across all sessions fetched, will
    # concat across animals and return
    animals_trials_df = []

    for animal_id in animal_ids:
        subject_session_key = {"ratname": animal_id}
        date_min_key = f"sessiondate >= '{date_min}'"
        date_max_key = f"sessiondate <= '{date_max}'"

        protocol_blobs = (
            bdata.Sessions & subject_session_key & date_min_key & date_max_key
        ).fetch("protocol_data", as_dict=True)

        # n session long items are fetched together
        sess_ids, dates, trials = (
            bdata.Sessions & subject_session_key & date_min_key & date_max_key
        ).fetch("sessid", "sessiondate", "n_done_trials")

        animals_trials_df.append(
            create_animals_trials_df(
                animal_id, protocol_blobs, sess_ids, dates, trials, verbose=verbose
            )
        )

    trials_df = pd.concat(animals_trials_df)

    return trials_df


def create_animals_trials_df(
    animal_id, protocol_blobs, sess_ids, dates, trials, verbose=True
):
    """
    TODO
    """
    protocol_blobs, sess_ids, dates, trials = drop_empty_sessions(
        protocol_blobs, sess_ids, dates, trials, verbose=verbose
    )

    protocol_dicts = convert_to_dicts(protocol_blobs)

    protocol_dfs = convert_to_dfs(protocol_dicts, sess_ids)

    append_and_clean_protocol_dfs(protocol_dfs, animal_id, sess_ids, dates, trials)

    animals_trials_df = pd.concat(protocol_dfs, ignore_index=True)

    print(
        f"fetched {len(dates)} sessions for {animal_id} between {min(dates)} and {max(dates)}"
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
        sess_dict = djut.transform_blob(session_blob[data_type])
        dicts.append(sess_dict)

    return dicts


def convert_to_dfs(dicts, sess_ids):
    """
    TODO
    """
    dfs = []

    for session_dict in dicts:
        try:
            dfs.append(djut.blob_dict_to_df(session_dict))
        except:
            print(f"error in fetching df for session {sess_ids}, skipping it")
            pass

    return dfs


def append_and_clean_protocol_dfs(dfs, animal_id, sess_ids, dates, trials):
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
        # sometimes lengths can get one off depending on when the
        # session ends in the PNT cycle, clipping the last row if needed
        if len(df) == n_done_trials + 1:
            df.drop(df.tail(1).index, inplace=True)

        assert (
            len(df) == n_done_trials
        ), f"session {sess_id} df is outside of length tolerance! trials: {n_done_trials} df: {len(df)}!"

        df.insert(0, "trial", np.arange(1, n_done_trials + 1, dtype=int))
        df.insert(1, "animal_id", [animal_id] * n_done_trials)
        df.insert(2, "date", [date] * n_done_trials)
        df.insert(3, "sess_id", [sess_id] * n_done_trials)

        # create a unique pair column indicating sa,sb (eg. "12-3")
        df["sound_pair"] = df.apply(
            lambda row: str(row.sa) + ", " + str(row.sb), axis=1
        )

        # convert data types (matlab makes everything a float)
        int_columns = ["result", "hits", "violations", "temperror"]
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
            "sound_pair",
            "first_spoke",
            "sides",
            "SMA_set",
            "go_type",
            "give_type_set",
            "give_type_imp",
        ]
        df[category_columns] = df[category_columns].astype("category")
