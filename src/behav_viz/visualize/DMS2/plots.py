"""
Author: Jess Breda
Date: July 23, 2024
Description: plots specific to the DMS2 protocol
both for within a day as well as across days. Many
of these are taken from plot_trials_info and plot_days_info
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu


######################################################################################
#########                        SINGLE DAY PLOTS                            #########
######################################################################################

############################ CENTER POKING STATS ######################################


def plot_cpoke_dur_over_trials(trials_df, ax=None, mode="settling_in", title=""):
    """
    Plot the animals time in the center port over trials with
    the required fixation marked in red.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `cpoke_dur` and `fixation_dur`
        with trials as row index
    ax : matplotlib.axes (optional, default = None)
        axis to plot to
    mode : str (default = "settling_in")
        whether the valid time is determined by settling_in + pre_go
        or just pre_go (when in "violations" mode)
    title : str, (optional, default = "")
        title of plot
    """

    if ax is None:
        _, ax = pu.make_fig()

    if mode == "settling_in":
        # if in this mode, violations don't exist yet. animal is in an early stage where
        # they need to poke for settling in dur length to start a trial and pre_go_dur
        # is something very small
        valid_time = trials_df.pre_go_dur + trials_df.settling_in_dur
    elif mode == "violations":
        # this is like a normal trial- settling in dur is baked into pre_dur and pre_go_dur
        # encapsulates all of the duration an animal should have fixated for
        valid_time = trials_df.pre_go_dur

    sns.lineplot(x="trial", y="cpoke_dur", data=trials_df, ax=ax, color="k")
    sns.lineplot(
        x="trial", y=valid_time, data=trials_df, ax=ax, color="red", label="valid time"
    )

    ax.set(xlabel="Trial", ylabel="Time in Cport [s]", title=title)
    ax.grid()

    return None


def plot_cpoke_dur_distributions(trials_df, ax, mode="settling_in", legend=False):
    """
    plot histogram of cpoke timing relative to the go cue for failed and valid
    cpokes across trials. Note that if mode is settling_in, a single trial can
    have both a successful and failed cpoke dur

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `avg_settling_in`, `cpoke_dur`,
        `pre_go_dur`, `settling_in_dur` `n_settling_ins` with trials as row index
    ax : matplotlib.axes
        axis to plot to
    mode : str (default = "settling_in")
        whether to plot the settling in dur or not. this is useful for the
        early stages where an animal must poke for settling in dur length
        to trigger go cue
    legend : bool, (default = False)
        whether to include legend or not
    """

    ## Curate Data
    data = pd.DataFrame()

    avg_cpoke_dur = trials_df.cpoke_dur.mean()
    if mode == "settling_in":
        # if in this mode, violations don't exist yet. animal is in an early stage where
        # they need to poke for settling in dur length to start a trial and pre_go_dur
        # is something very small
        valid_time = trials_df.pre_go_dur + trials_df.settling_in_dur

        data["failed_relative_to_go"] = trials_df.avg_settling_in - valid_time
        data["valid_relative_to_go"] = trials_df.cpoke_dur - valid_time

        ## Plot
        pal = [pu.RESULT_MAP[3]["color"], "lightgreen"]
        sns.histplot(data=data, binwidth=0.025, palette=pal, ax=ax)

        # vertical lines
        ax.axvline(0, color="k")
        ax.axvline(data.failed_relative_to_go.mean(), color=pal[0], lw=3)
        ax.axvline(data.valid_relative_to_go.mean(), color=pal[1], lw=3)

        # calculate how often animal needs multiple cpokes to start a trial
        multi_cpokes = np.sum(trials_df.n_settling_ins > 1) / len(trials_df)

        ax.set(
            xlabel="Cpoke Dur Relative to Go [s]",
            title=f"Failure rate {multi_cpokes:.2f}, Avg Cpoke Dur: {avg_cpoke_dur:.2f}",
        )

    elif mode == "violations":
        # this is like a normal trial- settling in dur is baked into pre_dur and pre_go_dur
        # encapsulates all of the duration an animal should have fixated for
        data["failed_settling_in"] = trials_df.avg_settling_in - trials_df.pre_go_dur
        data["cpoke_dur"] = trials_df["cpoke_dur"] - trials_df.pre_go_dur
        data["violated_trials"] = data["cpoke_dur"].where(
            trials_df["violations"] == 1, np.nan
        )
        data["valid_trials"] = data["cpoke_dur"].where(
            trials_df["violations"] == 0, np.nan
        )
        data.drop(columns=["cpoke_dur"], inplace=True)

        pal = ["blue", pu.RESULT_MAP[3]["color"], "lightgreen"]

        sns.histplot(data=data, binwidth=0.025, palette=pal, ax=ax)
        ax.axvline(0, color="k")

        ax.axvline(0, color="k")

        # below will break if valid or violated trials are all nan
        try:
            ax.axvline(data.violated_trials.mean(), color=pal[1], lw=3)
            ax.axvline(data.valid_trials.mean(), color=pal[2], lw=3)
        except:
            pass

        multi_cpokes = np.sum(trials_df.n_settling_ins > 1) / len(trials_df)
        viol_rate = np.sum(trials_df.violations) / len(trials_df)

        ax.set(
            xlabel="Cpoke Dur Relative to Go [s]",
            title=f"Mutli-Settling {multi_cpokes:.2f}, Viol Rate: {viol_rate:.2f},  Avg Cpoke Dur: {avg_cpoke_dur:.2f}",
        )

    return None


def plot_avg_failed_cpoke_dur(trials_df, ax, mode="settling_in"):
    """
    plot avg failed cpoke dur for per trial
    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `avg_settling_in`
        with trials as row index
    ax : matplotlib.axes
        axis to plot to
    """

    if mode == "settling_in":
        sns.histplot(
            trials_df.avg_settling_in,
            color=pu.RESULT_MAP[3]["color"],
            binwidth=0.025,
            ax=ax,
        )
        ax.axvline(
            trials_df.avg_settling_in.mean(),
            color=pu.RESULT_MAP[3]["color"],
            lw=3,
        )

        ax.set(
            xlabel="Failed Cpoke Dur [s]",
            title=f"Avg dur failed: {trials_df.avg_settling_in.mean():.2f}",
        )
    elif mode == "violations":
        data = pd.DataFrame()

        data["Settling"] = trials_df.avg_settling_in
        data["Viol"] = trials_df.query("violations ==1").cpoke_dur

        pal = ["blue", pu.RESULT_MAP[3]["color"]]

        sns.histplot(data=data, binwidth=0.025, palette=pal, ax=ax)

        avg_viol_cpoke = trials_df.query("violations == 1").cpoke_dur.mean()
        ax.axvline(avg_viol_cpoke, color=pal[1], lw=3)
        avg_settling_in = trials_df.avg_settling_in.mean()
        ax.axvline(avg_settling_in, color=pal[0], lw=3)
        ax.axvline(trials_df.pre_go_dur.mean(), color="k", lw=3)

        ax.set(
            xlabel="Failed Cpoke Dur [s]",
            title="Avg dur Viol {:.2f}".format(avg_viol_cpoke),
        )

    return None


def plot_avg_valid_cpoke_dur(trials_df, ax, mode="settling_in"):
    """
    plot avg valid cpoke dur for per trial

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `cpoke_dur`
        with trials as row index
    ax : matplotlib.axes
        axis to plot to
    """
    if np.sum(trials_df.cpoke_dur) == 0:
        print("No valid cpokes, make sure fixation is on!")
        return None

    sns.histplot(
        trials_df.query("violations == 0").cpoke_dur,
        color="lightgreen",
        binwidth=0.025,
        ax=ax,
    )

    avg_valid_cpoke = trials_df.query("violations == 0").cpoke_dur.mean()
    ax.axvline(avg_valid_cpoke, color="lightgreen", lw=3)

    if mode == "settling_in":
        valid_time = trials_df.pre_go_dur.mean() + trials_df.settling_in_dur.mean()
    elif mode == "violations":
        valid_time = trials_df.pre_go_dur.mean()
    ax.axvline(valid_time, color="k", lw=3)

    ax.set(
        xlabel="Valid Cpoke Dur [s]",
        title=f"Avg dur valid: {avg_valid_cpoke:.2f}",
    )

    return None


######################################################################################
#########                        MULTI DAY PLOTS                             #########
######################################################################################


############################ CENTER POKING STATS ######################################
