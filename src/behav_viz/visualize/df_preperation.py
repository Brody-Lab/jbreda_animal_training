"""
Author: Jess Breda
Date: July 22, 2024
Description: functions to prepare DataFrames for plotting
for plots that are general and can be used across protocols
"""

import pandas as pd


def rename_give_types(df):
    """
    Rename the give types in the dataframe to be
    shorter & more readable for plotting

    params:
    -------
    df: pandas.DataFrame
        dataframe with a column `give_type_imp` that
        contains the give type for a trial

    returns:
    --------
    pandas.Series
        series with the give types renamed
    """
    assert "give_type_imp" in df.columns, "give_type_imp not in df"
    mapping = {"water_and_light": "w + l", "water": "w", "light": "l", "none": "n"}
    return df.give_type_imp.replace(mapping)


def rename_curricula(df):
    """
    Truncate curriculum names for plotting
    params:
    -------
    df: pandas.DataFrame
        dataframe with a column `curriculum` that
        contains the curriculum name for a trial

    returns:
    --------
    pandas.Series
        series with the curriculum renamed
    """

    assert "curriculum" in df.columns, "curriculum not in df"
    # remove "JB_" prefix
    return df.curriculum.str.replace("JB_", "")


def compute_failed_fixation_rate_penalty_off(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the failed fixation rates for each animal_id, date
    in the dataframe. This will compute the failed fixation rate
    by trial and by poke when there is no violation penalty
    and animals get unlimited retries (ie.
    settling_in_determines_fixation is True).

    Specifically, by trial is the proportion of trials where
    the animal had more than one settling in. By poke is the
    proportion of total center pokes that were failed.

    For example, an animal could try 3 times to poke for a trial
    until successful. The first two pokes are failed, and the
    third is successful. This would be a 100% failed fixation
    rate by trial and a 66% failed fixation rate by poke.
    """

    def _compute_failed_fix_rates(group):

        # if there were no failed settling periods
        if (group["n_settling_ins"] > 1).sum() == 0:
            failed_fix_rate_by_trial = 0
            failed_fix_rate_by_poke = 0
        else:
            failed_fix_rate_by_trial = (group["n_settling_ins"] > 1).mean()
            failed_fix_rate_by_poke = 1 - (len(group) / group["n_settling_ins"].sum())
        return pd.Series(
            {
                "by_trial": failed_fix_rate_by_trial,
                "by_poke": failed_fix_rate_by_poke,
            }
        )

    # for each animal id, date, compute the failed fixation rates
    # by trial and by poke. Then, make a long dataframe with
    # animal_id, date, type (trial/poke), and failure_rate columns
    failed_fix_rates_df = (
        df.groupby(["animal_id", "date"])[df.columns]
        .apply(_compute_failed_fix_rates)
        .reset_index()
        .melt(
            id_vars=["animal_id", "date"],
            value_vars=["by_trial", "by_poke"],
            var_name="type",
            value_name="failure_rate",
        )
    )

    return failed_fix_rates_df


def make_long_trial_dur_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a long dataframe with animal_id, date, trial, type,
    and duration columns. The type column will be "Trial" or
    "ITI" and the durationcolumn will be the trial_dur or
    inter_trial_dur in seconds

    Trial relates to how long the trial actually took (inlcuding
    the ITI). The ITI is the time set by the SMA.
    """
    trial_dur_df = df.melt(
        id_vars=["animal_id", "date", "trial"],
        value_vars=["trial_dur", "inter_trial_dur"],
        var_name="type",
        value_name="duration",
    )

    trial_dur_df["type"] = trial_dur_df.type.replace(
        {"trial_dur": "Trial", "inter_trial_dur": "ITI"}
    )

    return trial_dur_df
