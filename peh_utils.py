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
    viol_state = trial_states["violation_state"]
    wf_spoke_state = trial_states["wait_for_spoke"]

    # print(wf_spoke_state)
    # check to see if trial was a violation
    # and if that violation was due to an early
    # side poke or no side poke
    if viol_state.size > 0:
        if wf_spoke_state.size > 0:
            pass  # violation due to no spoke
        else:
            # first entry into violation = time of early spoke
            spoke_in = viol_state[0]
    else:
        if wf_spoke_state.size > 2:
            # multi entry means Tup forgiveness was on,
            # grab last set & entry
            spoke_in = wf_spoke_state[-1][1]
        else:
            # single entry, grab the only out time
            spoke_in = wf_spoke_state[1]

    return spoke_in
