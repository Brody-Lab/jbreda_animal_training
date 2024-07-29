"""
Author: Jess Breda
Date: July 18, 2024
Description: plots specific to the FixationGrower protocol
both for within a day as well as across days
"""

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from behav_viz.utils import plot_utils as pu
from behav_viz.visualize.FixationGrower.df_preperation import (
    make_long_cpoking_stats_df,
)

######################################################################################
#########                        SINGLE DAY PLOTS                            #########
######################################################################################


############################ CENTER POKING STATS ######################################


def plot_avg_failed_cpoke_dur(trials_df: pd.DataFrame, ax: plt.Axes = None):
    """
    plot avg failed cpoke dur for per trial
    params
    ------
    trials_df : DataFrame
        trials dataframe with columns
        `avg_settling_in`, `cpoke_dur` and `violations`
        with trials as row index
    ax : matplotlib.axes, optional
        axis to plot to
    """

    if ax is None:
        _, ax = pu.make_fig("s")

    settling_in_mode = bool(trials_df.settling_in_determines_fixation.iloc[0])

    if settling_in_mode:  # Plot the failed settling in durs
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
    else:  # Plot the failed settling in durs (blue) and the failed cpoke, or violations (orange)
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


def plot_avg_valid_cpoke_dur(trials_df: pd.DataFrame, ax: plt.Axes = None):
    """
    plot avg valid cpoke dur for per trial

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns
        `cpoke_dur` and `violations`
        with trials as row index
    ax : matplotlib.axes, optional
        axis to plot to
    """

    if ax is None:
        _, ax = pu.make_fig("s")

    settling_in_mode = bool(trials_df.settling_in_determines_fixation.iloc[0])

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

    # TODO! if settling in and growth type is overnight, then we can do this
    # TODO! however, if growing each trial, this needs to be relative to go cue
    # TODO! which would be cpoke_dur - fixation_dur for the poking on
    # TODO! non violation trials and then 0 is the time of go cue

    valid_time = trials_df.fixation_dur.mean() + trials_df.settling_in_dur.mean()

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


def plot_cpoke_fix_stats_raw(
    trials_df,
    ax=None,
    title="Raw Fixation Stats",
    rotate_x_labels=True,
    legend=True,
):
    """
    Plot the raw fixation statistics over days. The dashed black
    line indicates the initial fixation duration for the animal.

    params
    ------
    trials_df : pd.DataFrame
        The trials dataframe with columns:
        - date
        - animal_id
        - trial
        - fixation_dur
        - cpoke_dur
        - violations
        - avg_settling_in
        - initial_fixation_dur
    ax : matplotlib.axes.Axes, optional (default=None)
        The axes to plot on.
    title : str, optional (default="Raw Fixation Stats")
        The title of the plot.
    rotate_x_labels : bool, optional (default=True)
        Whether to rotate the x-axis labels.
    legend : bool, optional (default=True)
        Whether to include the legend.

    """

    # compute center poking stats
    plot_df = make_long_cpoking_stats_df(trials_df, relative=False)

    if ax is None:
        fig, ax = pu.make_fig()

    ax.axhline(
        trials_df.initial_fixation_dur.mean(),
        color="black",
        linestyle="--",
    )
    sns.lineplot(
        data=plot_df,
        x="date",
        y="cpoke_dur",
        hue="was_valid",
        marker="o",
        hue_order=[True, False],
        palette=["green", "red"],
        ax=ax,
    )
    sns.lineplot(
        data=plot_df, x="date", y="fixation_dur", color="gray", marker=".", ax=ax
    )
    ax.axhline(
        trials_df.initial_fixation_dur.mean(),
        color="black",
        linestyle="--",
    )

    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    pu.set_legend(ax, legend)  # ax.legend(["Valid", "Invalid", "Fix", "Init"])
    if legend:
        ax.legend(title="Was Valid")
    _ = ax.set(title=title, ylabel="Duration [s]", xlabel="")

    return None


def plot_cpoke_fix_stats_relative(
    trials_df,
    ax=None,
    title="Relative Fixation Stats",
    rotate_x_labels=True,
    legend=True,
):
    """
    Plot the relative fixation statistics over days. Relative
    here means relative to the required fixation duration. This
    is helpful for curricula where the fixation duration changes
    on a trial-by-trial (rather than session) basis.

    params
    ------
    trials_df : pd.DataFrame
        The trials dataframe with columns:
        - date
        - animal_id
        - trial
        - fixation_dur
        - cpoke_dur
        - violations
        - avg_settling_in
    ax : matplotlib.axes.Axes, optional (default=None)
        The axes to plot on.
    title : str, optional (default="Raw Fixation Stats")
        The title of the plot.
    rotate_x_labels : bool, optional (default=True)
        Whether to rotate the x-axis labels.
    legend : bool, optional (default=True)
        Whether to include the legend.

    """

    # compute center poking stats
    plot_df = make_long_cpoking_stats_df(trials_df, relative=True)

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=plot_df,
        x="date",
        y="relative_cpoke_dur",
        hue="was_valid",
        marker="o",
        hue_order=[True, False],
        palette=["green", "red"],
        ax=ax,
    )
    sns.lineplot(
        data=plot_df,
        x="date",
        y="relative_fixation_dur",
        color="gray",
        marker=".",
        ax=ax,
    )

    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    pu.set_legend(ax, legend)  # ax.legend(["Valid", "Invalid", "Fix", "Init"])
    if legend:
        ax.legend(title="Was Valid")

    _ = ax.set(title=title, ylabel="Duration [s]", xlabel="")

    return None
