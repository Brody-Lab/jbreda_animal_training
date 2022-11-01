import numpy as np
from regex import P
import datajoint as dj
import blob_transformation as bt
from io_utils import *
import logging

dj.blob.use_32bit_dims = True
import pandas as pd
import blob_transformation as bt

log = logging.getLogger()
log.setLevel(logging.INFO)

## VARIABLES

MAP_SA_TO_SB = {
    12000: 3000,
    3000: 12000,
}

ANIMAL_IDS = ["R500", "R502", "R503", "R600"]


#### MULTI SESSION


def fetch_trials_table(animal_ids=None):

    animal_ids = ANIMAL_IDS if animal_ids is None else animal_ids
    assert type(animal_ids) == list, "animal ids must be in a list"

    animals_dfs = []
    bdata = dj.create_virtual_module("new_acquisition", "bdatatest")

    for animal_id in animal_ids:
        subject_session_key = {"ratname": animal_id}

        pd_blobs = (bdata.Sessions & subject_session_key).fetch(
            "protocol_data", as_dict=True
        )

        sess_ids, dates, trials = (bdata.Sessions & subject_session_key).fetch(
            "sessid", "sessiondate", "n_done_trials"
        )

        peh_blobs = (bdata.ParsedEvents * bdata.Sessions & subject_session_key).fetch(
            "peh", as_dict=True
        )

        trial_filter = (trials != 0) & (trials != 1)
        pd_blobs, peh_blobs, trials, sess_ids, dates = drop_sessions(
            pd_blobs, peh_blobs, trials, sess_ids, dates, trial_filter
        )

        pd_dicts = convert_to_dict(pd_blobs)
        reformat_pd_dict(pd_dicts)

        peh_dicts = convert_to_dict(peh_blobs)

        assert (
            len(pd_dicts)
            == len(peh_dicts)
            == len(trials)
            == len(sess_ids)
            == len(dates)
        ), "number of sessions does not match, assumptions of code below are broken!"

        peh_dicts, peh_dicts, trials, sess_ids, dates = find_and_fix_len_errors(
            pd_dicts, peh_dicts, trials, sess_ids, dates
        )

        peh_dicts_for_df = get_peh_vars_for_df(peh_dicts, trials, sess_ids)

        df = generate_trials_df(
            animal_id, pd_dicts, peh_dicts_for_df, trials, sess_ids, dates
        )

        print(
            f"fetched {len(dates)} sessions for {animal_id} with latest date {max(dates)}"
        )

        animals_dfs.append(df)

    all_animals_trials_table = pd.concat(animals_dfs, ignore_index=True)

    return all_animals_trials_table


# this is unique to trials_table
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


# this is used by both
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


# unique to trials table (similar to pd_prepare_dict_for_df)
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


def generate_trials_df(animal_id, pd_dicts, peh_dicts_for_df, trials, sess_ids, dates):
    # TODO doc string

    session_dfs = []

    # iterate over sessions & concat peh & pd trial by trial data into one df
    for pd_dict, peh_dict_for_df, n_done_trials, sess_id, date in zip(
        pd_dicts, peh_dicts_for_df, trials, sess_ids, dates
    ):
        pd_df = create_pd_df(pd_dict, animal_id, n_done_trials, sess_id, date)

        trials_df = concat_pd_and_peh_data(pd_df, peh_dict_for_df)

        session_dfs.append(trials_df)

    all_sessions_df = pd.concat(session_dfs, ignore_index=True)

    # add column for inferring go_time by adding fixation time to cpoke in
    # not using the 'go' wave here from peh because it will not be sent
    # on violation trials
    # TODO this could be a spot where you calculate additional variables & write
    # TODO a distinct function for it (e.g. trial length)
    all_sessions_df["go_time"] = all_sessions_df.apply(
        lambda row: row.cpoke_in + row.fixation, axis=1
    )

    return all_sessions_df


def create_pd_df(pd_dict, animal_id, n_done_trials, sess_id, date):
    # TODO doc strings
    pd_df = pd.DataFrame.from_dict(pd_dict)

    append_and_clean_pd_df(pd_df, animal_id, n_done_trials, sess_id, date)

    return pd_df


def concat_pd_and_peh_data(pd_df, peh_dict):
    # TODO doc strings

    peh_df = pd.DataFrame.from_dict(peh_dict)

    trials_df = pd.concat([pd_df, peh_df], axis=1)

    return trials_df


def append_and_clean_pd_df(pd_df, animal_id, n_done_trials, sess_id, date):
    """
    Function that takes a protocol_df and cleans it to correct for
    data types and format per JRBs preferences

    inputs
    TODO add updates
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

    # create trials column incase df index gets reset
    pd_df.insert(0, "trial", np.arange(1, n_done_trials + 1, dtype=int))

    # drop any trials where dispatcher reported a crash
    # this is where the remaining len errors between pd and peh
    # should be fixed
    pd_df.drop(pd_df[pd_df["result"] == 5].index, inplace=True)

    # add animal id, data and sessid value for each trial
    pd_df.insert(1, "animal_id", [animal_id] * len(pd_df))
    pd_df.insert(2, "date", [date] * len(pd_df))
    pd_df.insert(3, "sess_id", [sess_id] * len(pd_df))

    # convert units to kHz
    pd_df[["sa", "sb"]] = pd_df[["sa", "sb"]].apply(lambda row: row / 1000)
    # create a unique pair column & violation column used for sorting in plots
    pd_df["sound_pair"] = pd_df.apply(
        lambda row: str(row.sa) + ", " + str(row.sb), axis=1
    )
    pd_df["violations"] = pd_df.apply(lambda row: 1 if row.result == 3 else 0, axis=1)
    pd_df.insert(5, "violations", pd_df.pop("violations"))

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
    pd_df[int_columns] = pd_df[int_columns].astype("Int64")
    category_columns = ["result", "stage", "sess_id", "sound_pair"]
    pd_df[category_columns] = pd_df[category_columns].astype("category")

    ############
    ### PEH ####
    ############


def get_peh_vars_for_df(peh_dicts, trials, sess_ids):

    peh_trials_dicts = []
    for ii, (peh_dict, n_done_trials, sess_id) in enumerate(
        zip(peh_dicts, trials, sess_ids)
    ):

        # initialize vars of interest (will be columns of df)
        keys = ["early_spoke", "t_start", "t_end", "cpoke_in", "cpoke_out", "spoke_in"]
        peh_trials_dict = {key: (np.nan * np.ones((n_done_trials))) for key in keys}

        for it in range(n_done_trials):
            # renaming to keep the lines of code shorter
            trial_states = peh_dict[it]["states"]
            trial_pokes = peh_dict[it]["pokes"]

            # trial start and end time indicated by state 0 for all protocols
            try:
                peh_trials_dict["t_start"][it] = trial_states["state_0"][0][1]
                peh_trials_dict["t_end"][it] = trial_states["state_0"][1][0]
            except:
                print("tstart off")  # this is due to catching in a crash
                peh_trials_dict["t_start"][it] = np.nan
                peh_trials_dict["t_end"][it] = np.nan

            # cpoke that initiated the trial
            (
                peh_trials_dict["cpoke_in"][it],
                peh_trials_dict["cpoke_out"][it],
            ) = fetch_trial_start_cpoke(trial_states)

            # first spoke after trial initialization & if it was valid
            (
                peh_trials_dict["spoke_in"][it],
                peh_trials_dict["early_spoke"][it],
            ) = fetch_first_spoke(trial_states)

        peh_trials_dict["early_spoke"] = peh_trials_dict["early_spoke"].astype(bool)
        peh_trials_dicts.append(peh_trials_dict)

    return peh_trials_dicts


def fetch_trial_start_cpoke(trial_states):

    """
    Function to get the center poke time that initiates a trial.
    Since animal can enter cpoke state multiple times if they
    do not cross the 'settling_in_dur'. This changes the dims of the
    parsed events history for that trial such that:
    one entry = [in_time, out_time]
    multi entry = [[i_t, o_t], [i_t, o_t]]

    params
    ------
    trial_states : dict
        peh_dict['states'] dict for a single trial that contains
        all the state keys & timing values

    returns
    -------
    cpoke_in : int, seconds
        in time of c_poke state that started the trial in seconds
    cpoke_out : int, seconds
        out time of c_poke state that started the trial in seconds,
        also represent when the animal stopped poking in center
    """
    # early stages may just be side poking
    if "cpoke" not in trial_states:
        cpoke_in = np.nan
        cpoke_out = np.nan
    else:
        cpoke_state = trial_states["cpoke"]
        if cpoke_state.size > 2:
            # multi entry, grab times of last set
            cpoke_in = cpoke_state[-1][0]
            cpoke_out = cpoke_state[-1][1]
        else:
            # single entry, grab the only in time
            cpoke_in = cpoke_state[0]
            cpoke_out = cpoke_state[1]
    return cpoke_in, cpoke_out


def fetch_first_spoke(trial_states):
    """
    Function to determine the time of first side poke by grabbing
    the out time of wait_for_spoke since this state can only be
    exited if a side poke happens or if animal does not answer
    and state times up.

    If multiple entries in wait_for_spoke, Tup forgiveness was on
    and the final entry indicates a spoke. If only one entry,
    will check to see if violation state is empty and if so, final
    entry indicates a poke. If not, final entry is Tup time and no
    spoke happened on trial.

    params
    ------
    trial_states : dict
        peh_dict['states'] dict for a single trial that contains
        all the state keys & timing values

    returns
    -------
    spoke_in : int, seconds
        time of first side poke after trial starts, nan if no side poke was given
    """

    # rename for ease
    wf_spoke_state = trial_states["wait_for_spoke"]
    # early spoke state only assembled if viol penalty is off
    if "early_spoke_state" in trial_states:
        early_spoke_state = trial_states["early_spoke_state"]
    else:
        early_spoke_state = np.array([])
    # viol penalty only assembled when on cpoke init
    if "violation_state" in trial_states:
        viol_state = trial_states["violation_state"]
    else:
        viol_state = np.array([])

    # if animal made it to wf_spoke_state, they
    # didn't trigger a violation penalty during the
    # stimulus presentation. This is either because
    # the violation penalty is off (and an early poke
    # is documented under 'early_spoke_state') or
    # because they actually did the full trial
    if wf_spoke_state.size > 0:
        # if wfspoke & violation states both exist
        # the animal never spoked & wfspok Tuped to
        # violation state
        if viol_state.size > 0:
            spoke_in = np.nan
            early_spoke = False
            # print("no spoke")

        # if early spoke state exists, it is the
        # time of the first spoke
        elif early_spoke_state.size > 0:
            early_spoke = True
            if early_spoke_state.size > 2:
                spoke_in = early_spoke_state[-1][0]
            elif early_spoke_state.size == 2:
                spoke_in = early_spoke_state[1]

        # rest of options means the animal did
        # not violate or poke early, so we take
        # the final entry in wf_spoke because that
        # indicates when animal answered
        elif wf_spoke_state.size > 2:
            # multiple entries into wf_spoke
            # happens when Tup forgivness is on and
            # trial restarts if no answer was given
            spoke_in = wf_spoke_state[-1][1]
            # print("multi spoke")
            early_spoke = False
            # print("valid multi trial")

        elif wf_spoke_state.size == 2:
            # single entry into wf_spoke
            spoke_in = wf_spoke_state[1]
            early_spoke = False
            # print("valid trial")

    # animal violated due to an early spoke
    elif viol_state.size > 0:
        spoke_in = viol_state[0]
        early_spoke = True
        # log.info("violation penalty")

    else:
        print("trial & state structure unknown, cant det spoke time")

    return spoke_in, early_spoke
