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
        protocol_blobs, _, sess_ids, dates, trials = drop_empty_sessions(
            protocol_blobs, protocol_blobs, sess_ids, dates, trials
        )

        protocol_dicts = convert_to_dict(protocol_blobs)
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


def drop_empty_sessions(pd_blobs, peh_blobs, sess_ids, dates, trials):
    """
    sessions with 0 or 1 trials break the later code because
    of dimension errors, so drop them asap
    """
    # TODO replace with more flexible drop_sessions

    trial_filter = (trials != 0) & (trials != 1)
    print(
        f"dropping {len(pd_blobs) - np.sum(trial_filter)} sessions of {len(pd_blobs)}"
    )
    pd_blobs = np.array(pd_blobs)
    pd_blobs = pd_blobs[trial_filter]

    peh_blobs = np.array(peh_blobs)
    peh_blobs = peh_blobs[trial_filter]

    sess_ids = sess_ids[trial_filter]
    dates = dates[trial_filter]
    trials = trials[trial_filter]

    return pd_blobs, peh_blobs, sess_ids, dates, trials


def drop_sessions(
    protocol_data, parsed_events_hist, trials, sess_ids, dates, sess_to_keep
):
    inputs = [protocol_data, parsed_events_hist, trials, sess_ids, dates, sess_to_keep]
    assert (
        len({len(i) for i in inputs}) == 1
    ), "all inputs need to be of the same length!"

    # TODO log drops
    # TODO as a mesage input like "low trials", or "len mismatch"
    print(
        f"dropping {len(protocol_data) - np.sum(sess_to_keep)} sessions of {len(protocol_data)}"
    )
    protocol_data = np.array(protocol_data)
    protocol_data = protocol_data[sess_to_keep]

    parsed_events_hist = np.array(parsed_events_hist, dtype=object)
    parsed_events_hist = parsed_events_hist[sess_to_keep]

    trials = trials[sess_to_keep]
    sess_ids = sess_ids[sess_to_keep]
    dates = dates[sess_to_keep]

    return (
        protocol_data,
        parsed_events_hist,
        trials,
        sess_ids,
        dates,
    )


def convert_to_dict(blobs):
    """
    Function that takes protocol data (pd) or parsed events
    history (peh) blobs from bdata sessions table query and
    converts them to python dictionaries using Alvaro's blob
    transformation code

    inputs
    ------
    blobs : list of arrays
        list returned when fetching pd or peh data from bdata
        tables with len = n sessions

    returns
    -------
    dicts : list of dictionaries
        list of pd or peh dictionaries where len = n sessions and
        each session has a dictionary
    """
    # type of blob is indicated in nested, messy array structure
    data_type = list(blobs[0].keys())[0]
    assert data_type == "peh" or data_type == "protocol_data", "unknown key pair"

    dicts = []
    for session_blob in blobs:
        dict = bt.transform_blob(session_blob[data_type])
        dicts.append(dict)
    return dicts


def reformat_pd_dict(pd_dicts):
    """
    Quick function for updating 'sides' key to be
    properly seen as a list and making adding a more
    informative name to the 'match' bool. Performs
    formatting in-place.
    """
    # * if pd variables are ever added, this would be a good place
    # * to pack the sessions pre-dating the add w/ NaNs
    for pd_dict in pd_dicts:
        # lllrllr to [l, l, l, r....]
        pd_dict["sides"] = list(pd_dict["sides"])

        # if DMS, convert match/nonmatch category variable to bool
        # with more informative name
        if "dms_type" in pd_dict:
            pd_dict["is_match"] = pd_dict.pop("dms_type")
            pd_dict["is_match"] = pd_dict["is_match"].astype(bool)


def find_and_fix_len_errors(pd_dicts, peh_dicts, trials, sess_ids, dates):

    pd_has_extra_trial = []
    keep_session = []

    for pd_dict, peh_dict, n_done_trials, sess_id in zip(
        pd_dicts, peh_dicts, trials, sess_ids
    ):
        # TODO make these iterate for you!
        # make sure each pd dict variable is the same length
        # in response to some initial uncaught bugs in HistorySection.m
        fix_within_pd_len_errors(pd_dict, sess_id)

        # check if the number of trials in a session is the same between
        # all items for that session, and determine if it's fixable or
        # should be dropped.

        has_extra_trial, keep = find_and_assess_ntrial_mismatches(
            pd_dict, peh_dict, n_done_trials, sess_id
        )
        pd_has_extra_trial.append(has_extra_trial)
        keep_session.append(keep)

    # fix sessions that have one extra trial in pd data
    drop_extra_trial(pd_dicts, pd_has_extra_trial, trials)

    # drop sessions w/ atypical trial mismatches
    pd_dicts, peh_dicts, trials, sess_ids, dates = drop_sessions(
        pd_dicts, peh_dicts, trials, sess_ids, dates, sess_to_keep=keep_session
    )

    # TODO once iterator is fixed recall find_and_assess_ntrial_mismatches
    # and make sure np.sum(has_extra_trial) == 0 and np.sum keep_session = has no 0s

    return pd_dicts, peh_dicts, trials, sess_ids, dates


def fix_within_pd_len_errors(pd_dict, sess_id):
    """
    Function that fixes errors in protocol data dict due to keys
    having values of differing lengths when they should all be n
    trials long. Asserts lengths are the same before returning.

    inputs
    -----
    pd_dict : dict
        a single session's protocol data
    sess_id : int
        session id corresponding to the protocol data

    modifies
    --------
    pd_dict : dict
        modifies length of "sb" and "result" values in place if
        necessary & asserts all key lengths are the same
    """
    # TODO make this iterate over sessions w/i the functions
    # ! this only applies to DMS/PWM2 pre HistorySection bug fixes
    # ! using sa as reference length template- this could lead to
    # ! errors if sa length is incorrect
    if len(pd_dict["sa"]) != len(pd_dict["sb"]):
        _truncate_sb_length(pd_dict)
    if len(pd_dict["sa"]) != len(pd_dict["result"]):
        _fill_result_post_crash(pd_dict)

    # verify all keys have the same lengths
    lens = map(len, pd_dict.values())
    n_unique_lens = len(set(lens))
    assert n_unique_lens == 1, f"length of dict values for {sess_id} unequal!"


def find_and_assess_ntrial_mismatches(pd_dict, peh_dict, n_done_trials, sess_id):
    # TODO make this iterate over sessions w/i in the function
    # rename for ease
    pd_trials = len(pd_dict["sides"])
    peh_trials = len(peh_dict)

    # peh and n_done_trials are both updated by dispatcher once
    # a trial is completed, something is wrong if they aren't the same len
    assert (
        peh_trials == n_done_trials
    ), f"""sess {sess_id} peh trials: {peh_trials} 
            does not match n trials: {n_done_trials}"""

    # pd is updated before peh & n_done_trials and can sometimes
    # be 1 trial longer if the session is ended between these steps.

    # Or, if runrats freezes it will try to restart 10 times
    # before crashing, and pd will be 9 trials longer. This
    # is marked in pd['results'] and will be filtered out of the
    # pd_df before merging with peh data.

    # Other length differences are inconsistent & rare so I just
    # drop those sessions

    n_trials_extra = pd_trials - n_done_trials
    if n_trials_extra == 0 or n_trials_extra == 9:
        session_has_extra_trial = False
        keep = True
    elif n_trials_extra == 1:
        session_has_extra_trial = True
        keep = True
        # TODO log print(f"{sess_id} 1 off")
    else:
        session_has_extra_trial = False
        keep = False
        # TODO log print(f"{sess_id} being dropped bc {n_trials_extra}")

    return session_has_extra_trial, keep


def drop_extra_trial(pd_dicts, pd_has_extra_trial, trials):
    """
    Function to drop last trial from protocol data for a session
    if the number of completed trails differs from the number of
    started trials (pd tracks started, peh & trials tracks completed)

    inputs
    -----
    pd_dicts : list of dicts
        dictionary of protocol data for each session stored in a list

    pd_has_extra_trial : bool
        TODO: where is this coming from? is this the right name?
    trials : list
        number of completed trials for each session stored in a list

    modifies:
    --------
    pd_dicts : list of dicts
        removes extra trial from each dict key in place

    """
    # print(len(pd_dicts), len(pd_has_extra_trial))
    assert (
        len(pd_dicts) == len(pd_has_extra_trial) == len(trials)
    ), "number of sessions does not match!"

    for isess, (pd_dict, n_trials) in enumerate(zip(pd_dicts, trials)):

        if pd_has_extra_trial[isess]:
            # only use the values from 0 : n completed trials
            # i.e. exclude the last value
            for key, value in pd_dict.items():
                pd_dict[key] = value[:n_trials]
        else:
            pass


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
    for isess, protocol_dict in enumerate(protocol_dicts):
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
