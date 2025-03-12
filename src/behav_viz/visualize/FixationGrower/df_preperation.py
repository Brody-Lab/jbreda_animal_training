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
import behav_viz.visualize as viz


def determine_settling_in_mode(df: pd.DataFrame):
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
    return bool(df.settling_in_determines_fixation.iloc[-1])


def make_long_cpoking_stats_df(
    trials_df: pd.DataFrame, relative: bool, relative_to_stage: int = 5
) -> pd.DataFrame:
    """
    Wrapper function to apply make_long_cpoking_stats_df on groups by date
    and concatenate the results.This allows for variaton in settling in
    determines fixation from day to day to be handled, such that the settings
    from the most recent day (last trial) don't apply to all days.
    """
    trials_df = viz.df_preperation.compute_days_relative_to_stage(
        trials_df, stage=relative_to_stage
    )
    # Group by date and apply the existing function
    grouped_results = trials_df.groupby(["date"], observed=True).apply(
        lambda group: compute_cpoke_stats(group, relative, relative_to_stage)
    )

    # Reset index to flatten the multi-index created by groupby
    combined_cpoke_durs_long = grouped_results.reset_index(drop=True)

    return combined_cpoke_durs_long


def compute_cpoke_stats(
    trials_df: pd.DataFrame, relative: bool, relative_to_stage: int = 5
) -> pd.DataFrame:
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

    # Add the fixation_dur and relative to stage column from the original dataframe
    combined_cpoke_durs = combined_cpoke_durs.merge(
        trials_df[
            [
                "animal_id",
                "date",
                "trial",
                "fixation_dur",
                f"days_relative_to_stage_{relative_to_stage}",
            ]
        ],
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


def make_fixation_delta_df(df: pd.DataFrame, relative_stage: int = 5) -> pd.DataFrame:
    """
    Make a dataframe with the max fixation duration for each animal_id, date
    and the change in fixation duration from the previous day
    """

    df = viz.df_preperation.compute_days_relative_to_stage(df, stage=relative_stage)

    max_fixation_df = (
        df.query("stage >=5")  # only look at cpoking stages
        .groupby(
            [
                "date",
                "animal_id",
                "stage",
                f"days_relative_to_stage_{relative_stage}",
                "fix_experiment",
            ],
            observed=True,
        )
        .agg(
            max_fixation_dur=("fixation_dur", "max"),
            trials=("trial", "nunique"),
            n_violations=("violations", "sum"),
            n_settling_ins=("n_settling_ins", "sum"),
        )
        .reset_index()
    )
    max_fixation_df["valid_trials"] = (
        max_fixation_df["trials"] - max_fixation_df["n_violations"]
    )
    max_fixation_df.drop(columns=["n_violations"], inplace=True)

    # Compute the difference in fixation duration from the previous day
    max_fixation_df["fixation_delta"] = max_fixation_df.groupby(
        "animal_id"
    ).max_fixation_dur.diff()

    return max_fixation_df


def compute_failed_fixation_rate_df(df: pd.DataFrame) -> pd.DataFrame:
    # Group by date and apply the existing function
    grouped_results = df.groupby(
        ["animal_id", "date", "stage", "fix_experiment"], observed=False
    ).apply(compute_failed_fixation_rate)

    # Reset index to flatten the multi-index created by groupby
    failed_fix_df = grouped_results.reset_index(drop=True)

    return failed_fix_df


def compute_failed_fixation_rate(df: pd.DataFrame) -> pd.DataFrame:

    # determine penalty type- if settling in determines fix,
    # violation penalty is off. If not, violation penalty is on
    settling_in_determines_fix = determine_settling_in_mode(df)

    if settling_in_determines_fix:
        return compute_failed_fixation_rate_penalty_off(df)
    else:
        return compute_failed_fixation_rate_penalty_on(df)


def compute_failed_fixation_rate_penalty_off(group: pd.DataFrame) -> pd.DataFrame:
    # if there were no failed settling periods
    if (group["n_settling_ins"] > 1).sum() == 0:
        failed_fix_rate_by_trial = 0
        failed_fix_rate_by_poke = 0
    else:
        failed_fix_rate_by_trial = (group["n_settling_ins"] > 1).mean()
        failed_fix_rate_by_poke = 1 - (len(group) / group["n_settling_ins"].sum())
    return pd.DataFrame(
        {
            "animal_id": [group.animal_id.iloc[0]] * 2,
            "stage": [group.stage.iloc[0]] * 2,
            "fix_experiment": [group.fix_experiment.iloc[0]] * 2,
            "date": [group.date.iloc[0]] * 2,
            "type": ["by_trial", "by_poke"],
            "failure_rate": [failed_fix_rate_by_trial, failed_fix_rate_by_poke],
        }
    )


def compute_failed_fixation_rate_penalty_on(group: pd.DataFrame) -> pd.DataFrame:

    # take the violation rate for the date
    return pd.DataFrame(
        {
            "animal_id": [group.animal_id.iloc[0]],
            "stage": [group.stage.iloc[0]],
            "fix_experiment": [group.fix_experiment.iloc[0]],
            "date": [group.date.iloc[0]],
            "type": ["violation"],
            "failure_rate": [group.violations.mean()],
        }
    )


def filter_failed_fix_df(df: pd.DataFrame, min_stage, max_stage, settling_in_type):
    """
    Used for multi animal plots to filter the failed fixation rate dataframe
    """
    if min_stage is not None:
        df = df[df["stage"] >= min_stage]
    if max_stage is not None:
        df = df[df["stage"] <= max_stage]
    if settling_in_type == "by_poke":
        df = df.query("type != 'by_trial'")
    elif settling_in_type == "by_trial":
        df = df.query("type != 'by_poke'")
    return df


def compute_days_to_target_fix_df(
    df: pd.DataFrame, relative_stage: int = 5
) -> pd.DataFrame:
    """
    Computes the minimum days relative to a given stage for animals that have reached target fixation.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the trial data. Must include columns 'has_reached_target_fixation',
        'animal_id', 'fix_experiment', and 'days_relative_to_stage_{relative_stage}'.
    relative_stage : int, optional
        The stage relative to which the days are computed, by default 5.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns 'animal_id', 'fix_experiment', and 'days_to_target'.
    """
    df = viz.df_preperation.compute_days_relative_to_stage(df, stage=relative_stage)

    target_fix_df = (
        df.query("has_reached_target_fixation == True")
        .groupby(["animal_id", "fix_experiment"], observed=True)[
            f"days_relative_to_stage_{relative_stage}"
        ]
        .min()
        .reset_index()
    )

    target_fix_df.rename(
        columns={f"days_relative_to_stage_{relative_stage}": "days_to_target"},
        inplace=True,
    )

    return target_fix_df


def create_fixed_growth_stats_df(df, relative_stage):
    """Create a fixed growth statistics DataFrame.
     This DataFrame contains statistics such as the maximum fixation duration,
     number of trials, number of violations, and warm-up status for each training
     session. It is used as input for simulating fixed growth later.


    Params
    ------
    df (pandas.DataFrame):
        The input DataFrame containing the training session data.
    relative_stage (int):
        The stage relative to which the statistics should be calculated.

    Returns:
    -------
    pandas.DataFrame:
        The fixed growth statistics DataFrame.
    """
    fixed_growth_stats_df = (
        df.groupby(
            [
                "date",
                "animal_id",
                "stage",
                f"days_relative_to_stage_{relative_stage}",
            ]
        )
        .agg(
            max_fixation_dur=("fixation_dur", "max"),
            n_trials=("trial", "nunique"),
            n_violations=("violations", "sum"),
            warm_up_on=("delay_warm_up_on", "max"),
        )
        .reset_index()
    )

    fixed_growth_stats_df["violation_rate"] = (
        fixed_growth_stats_df["n_violations"] / fixed_growth_stats_df["n_trials"]
    )
    fixed_growth_stats_df["prev_day_max_fixation_dur"] = fixed_growth_stats_df.groupby(
        "animal_id"
    )["max_fixation_dur"].shift()

    return fixed_growth_stats_df


def simulate_fixed_growth_data(
    df, relative_stage, viol_rate=0, min_step_s=0.001, growth_frac=0.001
):
    """
    Simulate fixed growth for each training session in the input DataFrame.
    The simulated growth is stored in a new column called 'simulated_growth'.
    The specific use case of the function is to see how much an animal *could*
    grow in a session given a certain growth rate and violation rate.

    Params
    ------
    df (pandas.DataFrame):
        The input DataFrame containing the trial-by-trial data for each training
        session for multiple animals.
    relative_stage (int):
        The stage relative to which the statistics should be calculated.

    """

    fixed_growth_df = viz.df_preperation.compute_days_relative_to_stage(
        df.query("stage >= 5 and fix_experiment == 'V1'").copy(),
        stage=relative_stage,
    )

    fixed_growth_stats_df = create_fixed_growth_stats_df(
        fixed_growth_df, relative_stage
    )

    for idx, row in fixed_growth_stats_df.iterrows():
        starting_value = row.prev_day_max_fixation_dur

        growth_trials = row.n_trials - (20 * int(row.warm_up_on))
        simulated_growth_max, _ = simulate_fixed_growth(
            starting_value=starting_value,
            n_trials=growth_trials,
            viol_rate=viol_rate,
            min_step_s=min_step_s,
            growth_frac=growth_frac,
        )
        fixed_growth_stats_df.loc[idx, "simulated_max_fixation_dur"] = (
            simulated_growth_max
        )

    return fixed_growth_stats_df


def simulate_fixed_growth(
    starting_value, n_trials, min_step_s=0.001, growth_frac=0.001, viol_rate=0
):
    """
    Simulated fixed growth for a session of trials (post-warm up) given
    growth rate parameters and violation rate.
    """
    fix_dur = starting_value
    fix_durs = [fix_dur]

    for t in range(n_trials):
        # Determine if violation occurred
        if np.random.rand() < viol_rate:
            growth_step = 0
        else:
            growth_step = max(min_step_s, fix_dur * growth_frac)

        fix_dur += growth_step
        fix_durs.append(fix_dur)

    return fix_dur, fix_durs


def simulate_fixed_growth_w_warm_up(
    n_trials,
    warm_up_target,
    warm_up_viol_rate,
    non_warm_up_viol_rate,
    starting_value=0.01,
    n_warm_up=20,
    min_step_s=0.001,
    growth_frac=0.001,
):
    """
    Simulate fixation duration growth over a session that includes a warm-up phase
    and a post-warm-up phase.

    Parameters
    ----------
    n_trials : int
        Total number of trials in the session (includes warm-up + non-warm-up).
    warm_up_target : float
        The target fixation duration to reach by the end of the warm-up phase.
    warm_up_viol_rate : float
        Probability of a violation during warm-up trials.
    non_warm_up_viol_rate : float
        Probability of a violation during non-warm-up trials.
    starting_value : float, optional
        Initial fixation duration (in seconds). Default is 0.01.
    n_warm_up : int, optional
        Number of warm-up trials. Default is 20.
    min_step_s : float, optional
        Minimum step size (in seconds) for growth in non-warm-up trials. Default is 0.001.
    growth_frac : float, optional
        Fraction of the current duration to use as a growth step if larger than min_step_s.
        Default is 0.001.

    Returns
    -------
    final_fix_dur : float
        The final fixation duration after all trials.
    fix_durs : list of float
        The fixation duration after each trial.
    """

    fix_dur = starting_value
    fix_durs = [fix_dur]

    if n_warm_up > 0:
        warm_up_step = (warm_up_target - starting_value) / n_warm_up
    else:
        warm_up_step = 0  # no warm up steps if n_warm_up=0

    # Warm-up phase
    for t in range(n_warm_up):
        # Determine if violation occurred
        if np.random.rand() < warm_up_viol_rate:
            growth_step = 0
        else:
            growth_step = warm_up_step

        fix_dur += growth_step
        fix_durs.append(fix_dur)

    # Non-warm-up phase
    for t in range(n_warm_up, n_trials):
        # Determine if violation occurred
        if np.random.rand() < non_warm_up_viol_rate:
            growth_step = 0
        else:
            # Growth step is the maximum of min_step_s or 0.1% of the current duration
            growth_step = max(min_step_s, fix_dur * growth_frac)

        fix_dur += growth_step
        fix_durs.append(fix_dur)

    return fix_dur, fix_durs


def simulate_fixed_growth_data_w_warm_up(
    df,
    relative_stage,
    warm_up_viol_rate,
    non_warm_up_viol_rate,
    starting_value=0.01,
    n_warm_up=20,
    min_step_s=0.001,
    growth_frac=0.001,
    max_days=14,
):
    """
    Simulate fixed growth for each training session, including handling the warm-up phase.
    We will compute a 'simulated_max_fixation_dur' for each session given the parameters.

    Steps:
    -----
    1. We prepare the data by computing days relative to a given stage and creating a
       fixed growth stats DataFrame which contains info per day.
    2. We iterate through each day for each animal in chronological order.
    3. For each day:
       - If warm_up is on (row.warm_up_on == 1):
         starting_value = 0.01
         warm_up_target = previous day's simulated_max_fixation_dur if available,
                          else fallback to prev_day_max_fixation_dur if that's not null.
         run simulate_fixed_growth_w_warm_up with n_warm_up=20 (default)
       - If warm_up is off:
         starting_value = 0.01
         run simulate_fixed_growth_w_warm_up with n_warm_up=0, effectively no warm-up phase.
    4. Store the result in 'simulated_max_fixation_dur' for that day.
    """

    # Prepare data
    fixed_growth_df = viz.df_preperation.compute_days_relative_to_stage(
        df.query("stage >= 5 and stage < 9and fix_experiment == 'V1'").copy(),
        stage=relative_stage,
    )

    fixed_growth_df = fixed_growth_df.query(
        f"days_relative_to_stage_{relative_stage} < {max_days}"
    )

    fixed_growth_stats_df = create_fixed_growth_stats_df(
        fixed_growth_df, relative_stage
    )

    # Sort by animal_id and day relative to stage to ensure proper chronological order
    fixed_growth_stats_df = fixed_growth_stats_df.sort_values(
        by=["animal_id", f"days_relative_to_stage_{relative_stage}"]
    ).reset_index(drop=True)

    # We will store the simulated values here
    simulated_values = []

    # Keep track of previous day's simulated max per animal
    prev_simulated_max = {}

    for idx, row in fixed_growth_stats_df.iterrows():
        animal_id = row.animal_id

        # Determine if warm-up is on
        warm_up_on = bool(row.warm_up_on)

        # If warm-up is on, we need a warm-up target
        if warm_up_on:
            # Try to get warm-up target from previous day's simulated max for this animal
            if animal_id in prev_simulated_max and not pd.isnull(
                prev_simulated_max[animal_id]
            ):
                warm_up_target = prev_simulated_max[animal_id]
            else:
                # Fallback: if we have no previous simulated max, use prev_day_max_fixation_dur
                if not pd.isnull(row.prev_day_max_fixation_dur):
                    warm_up_target = row.prev_day_max_fixation_dur
                else:
                    # If no previous day data is available at all, fallback to current day's max?
                    # This scenario can happen if this is the animal's first day.
                    # For a first day, warm-up might not make sense, but we can at least do something reasonable:
                    warm_up_target = row.max_fixation_dur
        else:
            # If warm-up is off, we don't need a warm-up target. Just set it equal to starting_value or something neutral.
            warm_up_target = 0.01  # This won't matter since no warm-up trials will run.

        # Number of trials is row.n_trials
        n_trials = row.n_trials

        # Simulate
        final_fix_dur, fix_durs = simulate_fixed_growth_w_warm_up(
            n_trials=n_trials,
            warm_up_target=warm_up_target,
            warm_up_viol_rate=warm_up_viol_rate,
            non_warm_up_viol_rate=non_warm_up_viol_rate,
            starting_value=starting_value,
            n_warm_up=n_warm_up,
            min_step_s=min_step_s,
            growth_frac=growth_frac,
        )

        # The simulated maximum fixation duration is the max of fix_durs
        simulated_max_fixation_dur = max(fix_durs)

        # Store this in the DataFrame
        fixed_growth_stats_df.loc[idx, "simulated_max_fixation_dur"] = (
            simulated_max_fixation_dur
        )
        # Store the warm-up and non-warm-up violation rates
        fixed_growth_stats_df.loc[idx, "warm_up_viol_rate"] = warm_up_viol_rate
        fixed_growth_stats_df.loc[idx, "non_warm_up_viol_rate"] = non_warm_up_viol_rate

        # Update the prev_simulated_max for this animal
        prev_simulated_max[animal_id] = simulated_max_fixation_dur

    return fixed_growth_stats_df


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_v1_plot_data(fixed_growth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the DataFrame for plotting by computing deltas and melting into a long format.

    Parameters
    ----------
    fixed_growth_df : pd.DataFrame
        The DataFrame with simulated growth data including 'simulated_max_fixation_dur',
        'prev_day_max_fixation_dur', and 'max_fixation_dur'.

    Returns
    -------
    pd.DataFrame
        A melted DataFrame (v1_plot_df) suitable for plotting lineplots.
    """

    # Compute the true delta from actual data
    fixed_growth_df["V1_true_delta"] = fixed_growth_df.groupby("animal_id")[
        "max_fixation_dur"
    ].diff()

    # Compute the simulated ceiling (difference from previous day's max)
    fixed_growth_df["V1_simulated_ceiling"] = (
        fixed_growth_df["simulated_max_fixation_dur"]
        - fixed_growth_df["prev_day_max_fixation_dur"]
    )

    # Melt into long format for seaborn lineplot
    v1_plot_df = fixed_growth_df.melt(
        id_vars=["animal_id", "days_relative_to_stage_5"],
        value_vars=["V1_true_delta", "V1_simulated_ceiling"],
        var_name="delta_type",
        value_name="fixation_delta",
    )
    return v1_plot_df


def plot_v1_fixation_growth(
    v1_plot_df: pd.DataFrame, fixed_growth_df: pd.DataFrame, ylim=(-0.5, None)
):
    """
    Plot the V1 fixation growth using the prepped plot DataFrame.

    Parameters
    ----------
    v1_plot_df : pd.DataFrame
        The DataFrame returned by prepare_v1_plot_data.
    fixed_growth_df : pd.DataFrame
        The original DataFrame used to extract warm-up and non-warm-up violation rates
        for title labeling or other annotation purposes
    ylim : tuple, optional
        The y-axis limits for the plot. default is (-0.5, None)
    """

    # Extract violation rates (assuming they're consistent across sessions)
    # If they vary, you may need another strategy.
    warm_up_viol_rate = fixed_growth_df["warm_up_viol_rate"].dropna().unique()
    non_warm_up_viol_rate = fixed_growth_df["non_warm_up_viol_rate"].dropna().unique()

    # Fallback if arrays have more than one element or empty
    wuvr = warm_up_viol_rate[0] if len(warm_up_viol_rate) > 0 else 0
    nwuvr = non_warm_up_viol_rate[0] if len(non_warm_up_viol_rate) > 0 else 0

    fig, ax = pu.make_fig()
    sns.lineplot(
        data=v1_plot_df,
        x="days_relative_to_stage_5",
        y="fixation_delta",
        style="delta_type",
        color=pu.ALPHA_V1_color,
        marker="o",
        ax=ax,
    )
    ax.axhline(0, color="k", lw=2)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set(
        xlabel="Days Relative to Stage 5",
        ylabel="Fixation Duration Delta (s)",
        title=f"V1 Fixation Growth Simulated Ceiling vs True Delta: WUV = {wuvr}, NWUV = {nwuvr}",
        ylim=ylim,
    )
    return fig, ax
