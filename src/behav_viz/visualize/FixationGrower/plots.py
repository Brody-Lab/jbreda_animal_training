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
    determine_settling_in_mode,
)

######################################################################################
#########                        SINGLE DAY PLOTS                            #########
######################################################################################


############################ CENTER POKING STATS ######################################


def plot_cpoke_dur_over_trials(trials_df, ax=None, title=""):
    """
    Plot the animals time in the center port over trials with
    the required fixation marked in red.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `cpoke_dur` and `fixation_dur`
        with trials as row index
    ax : matplotlib.axes, (optional, default = None)
        axis to plot to
    title : str, (optional, default = "")
        title of plot
    """

    if ax is None:
        _, ax = pu.make_fig()

    sns.lineplot(x="trial", y="cpoke_dur", data=trials_df, ax=ax, color="k", marker=".")
    sns.lineplot(
        x="trial",
        y="fixation_dur",
        data=trials_df,
        ax=ax,
        color="red",
        label="Required",
    )

    ax.set(xlabel="Trial", ylabel="Time in Cport [s]", title=title)
    ax.grid()

    return None


def plot_cpoke_dur_distributions(trials_df, ax=None):
    """
    plot histogram of cpoke timing relative to the go cue for failed and valid
    cpokes across trials. This plot accounts for the SMA logic being used
    (e.g. settling_in_determines_fixation) to determine which cpokes are valid
    and which are not.

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `avg_settling_in`, `cpoke_dur`,
        `pre_go_dur`, `settling_in_dur` `n_settling_ins` with trials as row index
    ax : matplotlib.axes (optional, default = None)
        axis to plot to
    """

    # Prepare Data
    plot_df = make_long_cpoking_stats_df(trials_df, relative=True)
    settling_in_determines_fix = determine_settling_in_mode(trials_df)

    # Settings
    if settling_in_determines_fix:
        pal = ["blue", "lightgreen"]
        failure_rate = np.sum(trials_df.n_settling_ins > 1) / len(trials_df)
    else:
        pal = ["orangered", "lightgreen"]
        failure_rate = trials_df.violations.mean()

    # Plot
    if ax is None:
        _, ax = pu.make_fig()

    sns.histplot(
        data=plot_df,
        x="relative_cpoke_dur",
        binwidth=0.025,
        hue="was_valid",
        palette=pal,
        ax=ax,
    )

    avg_cpoke_dur = plot_df.cpoke_dur.mean()
    avg_failed_dur = plot_df.query("was_valid == False").relative_cpoke_dur.mean()
    avg_valid_dur = plot_df.query("was_valid == True").relative_cpoke_dur.mean()

    ax.axvline(0, color="k")
    ax.axvline(avg_failed_dur, color=pal[0], lw=3)
    ax.axvline(avg_valid_dur, color=pal[1], lw=3)

    # aesthetics
    _ = ax.set(
        xlabel="Cpoke Dur Relative to Go [s]",
        title=f"Failure_rate {failure_rate:.2f},  Avg Cpoke Dur: {avg_cpoke_dur:.2f}",
    )

    return None


def plot_avg_failed_cpoke_dur(trials_df: pd.DataFrame, ax=None):
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

    settling_in_determines_fix = determine_settling_in_mode(trials_df)

    if settling_in_determines_fix:
        # Plot the avg failed cpoke which is the settling in dur
        sns.histplot(
            trials_df.avg_settling_in,
            color="blue",
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
            title=f"Avg dur Settling: {trials_df.avg_settling_in.mean():.2f}",
        )
    else:  # Plot the failed settling in durs (blue) and the failed cpoke, or violations (orange)
        data = pd.DataFrame()

        data["Settling"] = trials_df.avg_settling_in
        data["Viol"] = trials_df.query("violations ==1").cpoke_dur

        pal = ["blue", "orangered"]

        sns.histplot(data=data, binwidth=0.025, palette=pal, ax=ax)

        # Plot avg failed cpoke
        avg_viol_cpoke = trials_df.query("violations == 1").cpoke_dur.mean()
        ax.axvline(avg_viol_cpoke, color=pal[1], lw=3)

        # Plot avg failed settling in
        avg_settling_in = trials_df.avg_settling_in.mean()
        ax.axvline(avg_settling_in, color=pal[0], lw=3)

        ax.axvline(trials_df.fixation_dur.mean(), color="k", lw=3)

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

    if np.sum(trials_df.cpoke_dur) == 0:
        print("No valid cpokes, skipping plot!")
        return None

    sns.histplot(
        trials_df.query("violations == 0").cpoke_dur,
        color="lightgreen",
        binwidth=0.025,
        ax=ax,
    )

    # Plot the average valid time
    avg_valid_cpoke = trials_df.query("violations == 0").cpoke_dur.mean()
    ax.axvline(avg_valid_cpoke, color="lightgreen", lw=3)

    # Plot the required poke time
    ax.axvline(trials_df.fixation_dur.mean(), color="k", lw=3)

    ax.set(
        xlabel="Valid Cpoke Dur [s]",
        title=f"Avg dur Valid: {avg_valid_cpoke:.2f}",
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
