from multiprocessing.spawn import prepare
import numpy as np
from regex import P

# from blob_transformation import mymblob_to_dict
import datajoint as dj

dj.blob.use_32bit_dims = True
import pandas as pd
from pathlib import Path
from datetime import date


import blob_transformation as bt

## VARIABLES

MAP_SA_TO_SB = {
    12000: 3000,
    3000: 12000,
}

TRAINING_DATA_PATH = "X:\\jbreda\\animal_data\\training_data"

ANIMAL_IDS = ["R500", "R501", "R502", "R503", "R600"]

## FUNCTIONS
# TODO: write a add_history_info function to add things like choice, prev_choice, etc


def fetch_latest_training_data(
    animal_ids=None, save_out=False, crashed_trials_report=False, print_sess_id=False
):
    """
    Function to query bdata via datajoint to get trial by trial
    protocol data for a(n) animal(s), clean it, save out and return
    as a single pandas data frame.

    inputs
    ------
    animal_ids : list, optional
        animal(s) to query database with, (default = ANIMAL_IDS)
    save_out : bool, optional
        if df should be saved as CSV to TRAINING_DATA_PATH
    crashed_trials_report : bool TODO
    print_sess_id : bool TODO

    returns
    -------
    all_animals_protocol_df : data frame
        data frame containing protocol data for every session for
        every animal animal_ids
    """
    animal_ids = ANIMAL_IDS if animal_ids is None else animal_ids
    assert type(animal_ids) == list, "animal ids must be in a list"

    animals_protocol_dfs = []

    # currently using `bdatatest` table but this may change &
    # will need to f/u with alvaro luna on which table has
    # protocol data
    bdata = dj.create_virtual_module("new_acquisition", "bdatatest")

    # fetch data, clean it & convert to df for each animal
    for animal_id in animal_ids:
        subject_session_key = {"ratname": animal_id}

        # protocol data needs to be fetched on it's own from sessions table
        # since it's trial length, then you can fetch session level items
        protocol_blobs = (bdata.Sessions & subject_session_key).fetch(
            "protocol_data", as_dict=True
        )
        sess_ids, dates, trials = (bdata.Sessions & subject_session_key).fetch(
            "sessid", "sessiondate", "n_done_trials"
        )

        # remove sessions with 0 or 1 trials
        protocol_blobs, sess_ids, dates, trials = drop_empty_sessions(
            protocol_blobs, sess_ids, dates, trials
        )

        protocol_dicts = pd_convert_to_dict(protocol_blobs)
        pd_prepare_dict_for_df(protocol_dicts)

        # here is where peh data can be loaded in before the final function of
        # make trianing_df. This means that the protocol_dicts need to be cleaned
        # then the peh data loaded, converted, parsed
        # so a cleaned protocol dict and a cleaned peh_dict merged together will
        # be passed into the next function once they are validated to be the
        # same length

        protocol_df = make_protocol_df(
            protocol_dicts,
            animal_id,
            sess_ids,
            dates,
            crashed_trials_report,
            print_sess_id,
        )
        animals_protocol_dfs.append(protocol_df)
        # using dates because can have multiple sess_ids in one session
        print(
            f"fetched {len(dates)} sessions for {animal_id} with latest date {max(dates)}"
        )

    # concatenate across animals
    all_animals_protocol_df = pd.concat(animals_protocol_dfs, ignore_index=True)

    if save_out:
        date_today = date.today().strftime("%y%m%d")  # reformat to YYMMDD
        file_name = f"{date_today}_training_data.csv"
        all_animals_protocol_df.to_csv(Path(TRAINING_DATA_PATH, file_name))

    return all_animals_protocol_df


def drop_empty_sessions(protocol_blobs, sess_ids, dates, trials):
    """
    sessions with 0 or 1 trials break the later code because
    of dimension errors, so drop them asap
    """

    trial_filter = (trials != 0) & (trials != 1)
    protocol_blobs = np.array(protocol_blobs)
    protocol_blobs = protocol_blobs[trial_filter]
    sess_ids = sess_ids[trial_filter]
    dates = dates[trial_filter]
    trials = trials[trial_filter]

    return protocol_blobs, sess_ids, dates, trials


def pd_convert_to_dict(protocol_blobs):
    """
    Function that takes protocol_data blobs from bdata sessions
    table query and converts them to python dictionaries

    inputs
    ------
    protocol_blobs : list of arrays
        list returned when fetching protocol_data from Sessions table
        with len = n sessions

    returns
    -------
    protocol_dicts : list of dictionaries
        list of protocol_data dictionaries where len = n sessions and
        each session has a dictionary
    """
    protocol_dicts = []
    for session in protocol_blobs:
        dict = bt.transform_blob(session["protocol_data"])
        protocol_dicts.append(dict)
    return protocol_dicts


def make_protocol_df(
    protocol_dicts,
    animal_id,
    sess_ids,
    dates,
    crashed_trials_report,
    print_sess_id,
):
    """
    Converts

    inputs:
    -------
    protocol_dicts : list of dictionaries
        protocol_data dictionary for each session queried
    animal_id : str
        id of animal that protocol_data is associated with, note this
        function currently assumes 1 animal per query
    sess_ids : arr
        session ids fetched from sessions table that correspond to
        values in protocol_dicts
    dates : arr
        dates fetched from sessions table that correspond to
        values in protocol_dicts
    crashed_trials_report : bool TODO
    print_sess_id : bool TODO

    !note!
        pd data structure must be saved to sessions table with all
        features having same length (n_started_trials). see DMS or PWM2
        protocols' HistorySection.m for example in `make_and_send_summary`
        Early stages of testing these new protocols didn't follow this
        rule & caused bugs that are outlined & fixed via `truncate_sb_length`
        and `fill_results_post_crash`

    returns:
    -------
    all_sessions_protocol_df: date frame
        protocol_data for every session for an animal
    """
    sess_protocol_dfs = []

    # for each session, turn protocol data dict into data frame
    # and then concatenate together into single data frame
    for isess, sess_id in enumerate(sess_ids):
        if print_sess_id:
            print(f"preparing session id: {sess_id} for {animal_id}")

        # skip sessions with 0 or 1 trials bc they cause len problems in
        # later functions & aren't worth analyzing
        if len(protocol_dicts[isess]["sides"]) in [0, 1]:
            continue

        protocol_df = pd.DataFrame.from_dict(protocol_dicts[isess])
        clean_pd_df(
            protocol_df,
            animal_id,
            sess_id,
            dates[isess],
            crashed_trials_report,
        )
        sess_protocol_dfs.append(protocol_df)

    all_sessions_protocol_df = pd.concat(sess_protocol_dfs)
    return all_sessions_protocol_df


def pd_prepare_dict_for_df(protocol_dicts):
    """
    Function to clean up protocol data dictionary lengths,
    names & types to ensure there are no errors & interpretation
    issues upon converting it into a data frame.

    inputs
    ------
    protocol_dicts : list of dicts
        list of dictionaries for one or more sessions protocol data

    modifies
    --------
    protocol_dicts : list of dicts
        corrects side vector format, updates DMS match/nonmatch
        variable names, corrects for bugs found in HistorySection.m
        that led to differing variables lengths
    """
    for protocol_dict in protocol_dicts:
        # lllrllr to [l, l, l, r....]
        protocol_dict["sides"] = list(protocol_dict["sides"])

        # if DMS, convert match/nonmatch category variable to bool
        # with more informative name
        if "dms_type" in protocol_dict:
            protocol_dict["is_match"] = protocol_dict.pop("dms_type")
            protocol_dict["is_match"] = protocol_dict["is_match"].astype(bool)

        # check to see if protocol_data is pre HistorySection.m bug fixes
        # where len(each value) was not equal. Using sa as reference length
        # template but this could lead to errors if sa has bug
        if len(protocol_dict["sa"]) != len(protocol_dict["sb"]):
            _truncate_sb_length(protocol_dict)
        if len(protocol_dict["sa"]) != len(protocol_dict["result"]):
            _fill_result_post_crash(protocol_dict)

        # catch any remaining length errors
        lens = map(len, protocol_dict.values())
        n_unique_lens = len(set(lens))
        assert n_unique_lens == 1, "length of dict values unequal!"


def _truncate_sb_length(protocol_dict):
    """
    Function to correct for bug in HistorySection, see commit:
    https://github.com/Brody-Lab/Protocols/commit/4a2fadb802d64b7ed66891a263a366a8d2580483
    sb vector was 1 greater than n_started_trials due to an
    appending error

    inputs
    ------
    protocol_dict : dict
        dictionary for a single session's protocol_data

    modifies
    -------
    protocol_dict : dict
        updated protocol_dict with sb column length & contents corrected
        if DMS. length only corrected if PWM2 protocol is being used
    """

    # rename for ease
    sa = protocol_dict["sa"]
    sb = protocol_dict["sb"]
    match = protocol_dict["is_match"]

    # if DMS task, can infer values
    if "is_match" in protocol_dict:
        for trial in range(len(sa)):
            if match[trial]:
                sb[trial] = sa[trial]  # update sb
            else:
                assert sa[trial] in MAP_SA_TO_SB, "sa not known"
                sb[trial] = MAP_SA_TO_SB[sa[trial]]
    else:
        print("sb values incorrect, only fixing length")

    sb = sb[0:-1]  # remove extra entry
    protocol_dict["sb"] = sb  # update


def _fill_result_post_crash(protocol_dict):
    """
    Function to correct for bug in HistorySection, see commit:
    https://github.com/Brody-Lab/Protocols/commit/3bdde4377ffde011cc34d098acfeb77b74c9e606
    result vector was shorter than n_started_trials because program
    crashed & results_history vector was not being properly filled
    during crash clean up

    inputs
    ------
    protocol_dict : dict
        dictionary for a single session's protocol_data

    modifies
    -------
    protocol_dict : dict
        updated protocol_dict with results column length corrected to
        reflect crash trials
    """

    # rename for ease
    results = protocol_dict["result"]
    sa = protocol_dict["sa"]

    # pack with crash result value (5)
    crash_appended_results = np.ones((len(sa))) * 5
    crash_appended_results[0 : len(results)] = results

    protocol_dict["result"] = crash_appended_results


def clean_pd_df(protocol_df, animal_id, sess_id, date, crashed_trials_report=False):
    """
    Function that takes a protocol_df and cleans it to correct for
    data types and format per JRBs preferences

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

    n_started_trials = len(protocol_df)
    # create trials column incase df index gets reset
    protocol_df.insert(0, "trial", np.arange(1, n_started_trials + 1, dtype=int))

    # drop any trials where dispatcher reported a crash
    protocol_df.drop(protocol_df[protocol_df["result"] == 5].index, inplace=True)
    n_completed_trials = len(protocol_df)
    if crashed_trials_report:
        if n_started_trials > n_completed_trials:
            print(
                f"{n_started_trials - n_completed_trials} crash trials found for {animal_id} on {sess_id}"
            )

    # add animal id, data and sessid value for each trial
    protocol_df.insert(1, "animal_id", [animal_id] * len(protocol_df))
    protocol_df.insert(2, "date", [date] * len(protocol_df))
    protocol_df.insert(3, "sess_id", [sess_id] * len(protocol_df))

    # convert units to kHz
    protocol_df[["sa", "sb"]] = protocol_df[["sa", "sb"]].apply(lambda row: row / 1000)
    # create a unique pair column & violation column used for sorting in plots
    protocol_df["sound_pair"] = protocol_df.apply(
        lambda row: str(row.sa) + ", " + str(row.sb), axis=1
    )
    protocol_df["violations"] = protocol_df.apply(
        lambda row: 1 if row.result == 3 else 0, axis=1
    )
    protocol_df.insert(5, "violations", protocol_df.pop("violations"))

    # convert data types (matlab makes everything a float)
    int_columns = [
        "hits",
        "violations",
        "temperror",
        "result",
        "helper",
        "stage",
        "sess_id",
    ]
    protocol_df[int_columns] = protocol_df[int_columns].astype("Int64")
    category_columns = ["result", "stage", "sess_id", "sound_pair"]
    protocol_df[category_columns] = protocol_df[category_columns].astype("category")
