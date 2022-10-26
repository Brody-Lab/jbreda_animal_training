import numpy as np
from regex import P
import datajoint as dj
import blob_transformation as bt
from io_utils import *
from peh_utils import *

dj.blob.use_32bit_dims = True
import pandas as pd
import blob_transformation as bt

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
