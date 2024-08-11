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


def compute_days_relative_to_stage(
    df: pd.DataFrame, stage: int, date_col_name: str = "date"
) -> pd.DataFrame:
    """
    Compute the number of days relative to a specific stage in the dataframe.

    params:
    -------
    df : pd.DataFrame
        DataFrame containing the data
    stage : int
        The specific stage to compute the relative days for
    date_col_name : str, optional
        The name of the column containing the dates, by default "date"

    returns:
    --------
    pd.DataFrame
    DataFrame with an additional column indicating the number of days relative to the stage
    """
    # convert to date time
    df["datetime_col"] = pd.to_datetime(df[date_col_name])

    # find first day in stage for each animal
    min_date_stage = (
        df.query("stage == @stage")
        .groupby("animal_id")["datetime_col"]
        .min()
        .reset_index()
    )
    min_date_stage.rename(
        columns={"datetime_col": f"min_date_stage_{stage}"}, inplace=True
    )

    # merge on animal_id & subtract min column wise
    df = df.merge(min_date_stage, on="animal_id", how="left")
    df[f"days_relative_to_stage_{stage}"] = (
        df["datetime_col"] - df[f"min_date_stage_{stage}"]
    ).dt.days

    df.drop(columns=["datetime_col", f"min_date_stage_{stage}"], inplace=True)

    return df


def make_days_in_stage_df(df, min_stage=None, max_stage=None, hue_var=None):
    """
    Compute the number of days spent in each stage, for each animal in the df

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    min_stage : int, optional
        The minimum stage value to include in the computation, by default None
    max_stage : int, optional
        The maximum stage value to include in the computation, by default None
    hue_var : str, optional
        Additional variable to use for grouping, by default None  and groups by
        animal_id, stage, and fix_experiment

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional column indicating the number of days relative to the stage
    """
    # query the df for stage >= min_stage and stage <= max_stage
    # if they are not None
    if min_stage is not None:
        df = df.query("stage >= @min_stage")
    if max_stage is not None:
        df = df.query("stage <= @max_stage")
    if hue_var is None:
        cols = ["animal_id", "stage", "fix_experiment"]
    else:
        cols = ["animal_id", "stage", "fix_experiment", hue_var]

    days_in_stage_df = (
        df.groupby(cols)
        .agg(n_days=pd.NamedAgg(column="date", aggfunc="nunique"))
        .reset_index()
    )

    return days_in_stage_df
