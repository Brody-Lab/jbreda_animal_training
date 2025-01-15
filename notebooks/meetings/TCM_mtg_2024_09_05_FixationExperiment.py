""" 
Python file to keep 2024 thesis committe meeting notebook clean(er)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import behav_viz.visualize as viz
import behav_viz.ingest as ingest
import behav_viz.visualize.FixationGrower as FG
from behav_viz.utils import plot_utils as pu

import sys

sys.path.append(
    "/Users/jessbreda/Desktop/github/jbreda_animal_training/notebooks/_active_experiments/FixationGrower"
)
import config_alpha_experiment as alpha_config

### DFs for Plots ###


def make_target_yield_df(aid_still_growing, aid_reached_target):

    v1_successes = set(aid_reached_target).intersection(set(alpha_config.ALPHA_V1))
    v2_successes = set(aid_reached_target).intersection(set(alpha_config.ALPHA_V2))
    v1_failures = set(aid_still_growing).intersection(set(alpha_config.ALPHA_V1))
    v2_failures = set(aid_still_growing).intersection(set(alpha_config.ALPHA_V2))

    conditions = ["V1", "V2"]
    success_sets = [v1_successes, v2_successes]
    failure_sets = [v1_failures, v2_failures]

    # Create a list of dictionaries to hold the data
    data = []
    success_rates = []
    for cond, successes, failures in zip(conditions, success_sets, failure_sets):
        n_success = len(successes)
        n_failure = len(failures)
        success_rate = (
            n_success / (n_success + n_failure) if (n_success + n_failure) > 0 else 0
        )
        success_rates.append(success_rate)
        data.append({"fix_experiment": cond, "outcome": "Succeed", "count": n_success})
        data.append({"fix_experiment": cond, "outcome": "Fail", "count": n_failure})

    target_yield_df = pd.DataFrame(data)

    return target_yield_df, success_rates


def make_still_growing_df(df, aid_still_growing):

    # compute days since starting stage 5
    still_growing_df = viz.df_preperation.compute_days_relative_to_stage(
        df.query("animal_id in @aid_still_growing and stage >=5").copy(), 5
    )

    # determine the day since they started & rename for merging
    still_growing_df = (
        still_growing_df.groupby(["animal_id", "fix_experiment"], observed=True)
        .days_relative_to_stage_5.max()
        .reset_index()
        .rename(columns={"days_relative_to_stage_5": "days"})
    )

    # indicate that they are still growing
    still_growing_df["reached_target"] = False

    return still_growing_df


def make_failed_fix_df_by_penalty_type(df):

    failed_fix_df = viz.FixationGrower.df_preperation.compute_failed_fixation_rate_df(
        df.query("stage >=5")
    )

    # remove by_trial data- we only want to use the by_poke metric for
    # a failed fixation- meaning of all the center pokes, how many were invalid?
    # this is larger than the by_trial metric which is how many trials had
    # more than 1 center poke
    failed_fix_df.query("type != 'by_trial'", inplace=True)

    type_map = {"violation": "Penalty On", "by_poke": "Penalty Off"}
    failed_fix_df["type"] = failed_fix_df.type.map(type_map)

    return failed_fix_df


### Plots ###
def box_swarm_v1_vs_v2(data, x, order, y, ax=None, **kwargs):
    """ """

    if ax is None:
        fig, ax = pu.make_fig("m")
    sns.despine()

    sns.boxplot(
        x=x,
        y=y,
        data=data,
        order=order,
        ax=ax,
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        fill=None,
        showfliers=False,
        dodge=True,
        **kwargs,
    )
    sns.swarmplot(
        x=x,
        y=y,
        data=data,
        order=order,
        ax=ax,
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        dodge=True,
        legend=False,
        alpha=0.5,
        **kwargs,
    )

    ax.legend(title=None, frameon=False)

    return fig, ax


def box_strip_v1_vs_v2(data, x, order, y, ax, **kwargs):
    """ """

    sns.despine()

    sns.boxplot(
        x=x,
        y=y,
        data=data,
        order=order,
        ax=ax,
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        fill=None,
        showfliers=False,
        dodge=True,
        **kwargs,
    )
    sns.stripplot(
        x=x,
        y=y,
        data=data,
        order=order,
        ax=ax,
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        dodge=True,
        legend=False,
        alpha=0.5,
        **kwargs,
    )

    ax.legend(title=None, frameon=False)


def plot_fixation_growth_over_trials(
    tdf, animal_id, n_days_back=None, ax=None, **kwargs
):
    # Filter the DataFrame for the specified animal_id
    animal_df = tdf.query("animal_id == @animal_id and stage >=5 and stage <=7").copy()

    # If n_days_back is provided, select the most recent n days
    if n_days_back is not None:
        days = animal_df.date.unique()[-n_days_back:]
        print(f"Analyzing data from the following days: {days}")
        animal_df = animal_df.query("date in @days").copy()

    # Add a cumulative trial column
    animal_df["cumulative_trial"] = range(len(animal_df))

    # Create the figure and axis if not provided
    if ax is None:
        fig, ax = pu.make_fig()

    # Create the line plot
    sns.lineplot(
        data=animal_df, x="cumulative_trial", y="fixation_dur", ax=ax, **kwargs
    )

    # Add a horizontal line at y=2
    ax.axhline(2, color="black")

    # Customize the plot aesthetics
    sns.despine()
    _ = ax.set(
        xlabel="Cumulative Trial",
        ylabel="Fixation Duration",
        title=f"{animal_id} Fixation Growth",
        ylim=(-0.1, 2.5),
    )

    return ax


#### Violation ITI Class ####
""" 
Class for computing violation ITI (inter-trial-interval) 
for a modern dataset. Additional methods allow for validation
of the computed ITI function with simulated data as well as
plotting. 

Note: a single object can be run on multiple datasets using
the computer_and_add_viol_iti_column method.

Modified from viol multi codebase by: Jess Breda 2024-08-04
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ViolationITI:
    def __init__(self):
        pass

    def compute_and_add_viol_iti_column(self, df):
        self.run_checks(df)

        df["violation_iti"] = np.nan
        df_with_iti = (
            df.groupby(["animal_id", "date"])
            .apply(self.calculate_violation_intervals)
            .reset_index(drop=True)
        )

        return df_with_iti

    def plot_data_over_trials(self, df, start_idx=None, end_idx=None):
        if start_idx:
            plot_data = df.iloc[start_idx:end_idx]
        else:
            plot_data = df.copy()

        n_dates = plot_data.session.nunique()

        fig, ax = plt.subplots(
            n_dates, 1, figsize=(10, (n_dates * 3)), constrained_layout=True
        )

        for i, (date, group) in enumerate(plot_data.groupby("date")):
            cur_ax = ax[i] if n_dates > 1 else ax
            cur_ax.plot(
                group.trial,
                group.violation,
                "o",
                label="Viol Hist",
            )
            cur_ax.plot(
                group.trial,
                group.violation_iti,
                "x",
                color="red",
                label="Viol ITI",
            )

            cur_ax.set(
                xlabel="Trial",
                ylabel="was viol / viol ITI",
                title=f"Date: {date}",
            )
        cur_ax.legend()

    @staticmethod
    def simulate_data(n_sessions=3, n_trials=25, random_state=None):
        data = []

        for session in range(1, n_sessions + 1):
            for trial in range(1, n_trials + 1):
                # Randomly decide if a violation occurred (for simplicity, say 20% chance)
                np.random.seed(random_state)
                violation = np.random.choice(
                    [0, 1],
                    p=[0.8, 0.2],
                )
                data.append(["example_animal", session, trial, violation])

        # Create DataFrame
        columns = ["animal_id", "date", "trial", "violations"]
        simulated_df = pd.DataFrame(data, columns=columns)

        return simulated_df

    @staticmethod
    def run_checks(df):
        required_columns = ["animal_id", "trial", "date", "violations"]
        assert all(
            column in df.columns for column in required_columns
        ), "Required columns not present!"

    @staticmethod
    def calculate_violation_intervals(group):
        """
        function to be used with a pandas "apply" when
        grouping by session (can also group by animal_id
        if running multiple animals at a time)
        """

        # Get trial numbers of trials with a violation
        if group.settling_in_determines_fixation.iloc[0]:
            violation_trials = group["trial"][group["n_settling_ins"] > 1]
        else:
            violation_trials = group["trial"][group["violations"] == 1]

        # Calculate number of trials between trials with a violation
        # shift by 1 to align trials properly
        intervals = violation_trials.diff() - 1

        # Assign the calculated intervals back to the group,
        # All violation trials (except the first of a group) get a value,
        # every other trial type gets an nan
        group.loc[violation_trials.index, "violation_iti"] = intervals

        return group


def plot_violation_iti(df, ax=None, stat="percent", binwidth=1, title="", **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    sns.histplot(
        data=df,
        x="violation_iti",
        stat=stat,
        ax=ax,
        binwidth=1,
        **kwargs,
    )

    ax.set(
        xlabel="Inter Violation Interval (N Trials)",
        ylabel="Percent of Trials",
        xlim=(0, 25),
        title=title,
    )

    color = kwargs.get("color", "black")

    median = df.violation_iti.median()
    ax.axvline(median, linestyle="--", color=color)

    ax.text(
        median + 0.5,
        0.95 * ax.get_ylim()[1],
        str(median),
        color=color,
    )

    sns.despine()

    return ax
