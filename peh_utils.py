import logging
import numpy as np

log = logging.getLogger()
log.setLevel(logging.INFO)


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


# def peh_prep_dict_for_df(peh_dict):
