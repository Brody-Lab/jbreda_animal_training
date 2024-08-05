"""
Author: Jess Breda
Date: July 22, 2024
Description: functions to prepare DataFrames for plotting
specific to FixationGrower plots
"""

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from behav_viz.utils import plot_utils as pu


def determine_settling_in_mode(trials_df: pd.DataFrame):
    """
    Determine if the settling in mode is being used based on the
    value of the settling_in_determines_fixation column in the trials_df

    ! Note assumes last row in trials df is representative of the current state

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `settling_in_determines_fixation`
        with trials as row index
    returns
    -------
        : bool
        whether the settling in determines fixation and violations are
        effectively impossible
    """
    return bool(trials_df.settling_in_determines_fixation.iloc[-1])


def make_long_cpoking_stats_df(trials_df: pd.DataFrame, relative: bool) -> pd.DataFrame:
    """
    Function to create a long dataframe of cpoking statistics for plotting
    with columns for animal_id, date, trial, cpoke_dur, was_valid, and fixation_dur

    was_valid indicates if it was a failed or valid center poke. Depending on the
    curriculum this could be a failed settling in or a violation
    """

    # Determine the settling_in_mode- this determines where the failed poking
    # information is stored
    settling_in_determines_fix = determine_settling_in_mode(trials_df)

    # Handle valid and non-valid cpoke durations based on the mode
    if settling_in_determines_fix:
        non_valid_cpoke_durs = trials_df[["animal_id", "date", "trial"]].copy()
        non_valid_cpoke_durs["cpoke_dur"] = trials_df.avg_settling_in
        non_valid_cpoke_durs["was_valid"] = False
    else:
        non_valid_cpoke_durs = trials_df.query("violations == 1")[
            ["animal_id", "date", "trial", "cpoke_dur"]
        ].copy()
        non_valid_cpoke_durs["was_valid"] = False

    valid_cpoke_durs = trials_df.query("violations == 0")[
        ["animal_id", "date", "trial", "cpoke_dur"]
    ].copy()

    valid_cpoke_durs["was_valid"] = True

    # Combine valid and non-valid cpoke durations into one long, dataframe
    combined_cpoke_durs = pd.concat(
        [valid_cpoke_durs, non_valid_cpoke_durs], ignore_index=True
    )

    # Add the fixation_dur column from the original dataframe
    combined_cpoke_durs = combined_cpoke_durs.merge(
        trials_df[["animal_id", "date", "trial", "fixation_dur"]],
        on=["animal_id", "date", "trial"],
        how="left",
    )

    if relative:
        # Calculate the relative fixation duration
        combined_cpoke_durs["relative_cpoke_dur"] = (
            combined_cpoke_durs["cpoke_dur"] - combined_cpoke_durs["fixation_dur"]
        )
        combined_cpoke_durs["relative_fixation_dur"] = (
            combined_cpoke_durs["fixation_dur"] - combined_cpoke_durs["fixation_dur"]
        )

    return combined_cpoke_durs


def make_fixation_delta_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a dataframe with the max fixation duration for each animal_id, date
    and the change in fixation duration from the previous day
    """
    max_fixation_df = (
        df.query("stage >=5")  # only look at cpoking stages
        .groupby(["date", "animal_id"])
        .fixation_dur.max()
        .reset_index()
    )
    max_fixation_df = max_fixation_df.rename(
        columns={"fixation_dur": "max_fixation_dur"}
    )
    max_fixation_df["fixation_delta"] = max_fixation_df.max_fixation_dur.diff()

    return max_fixation_df
