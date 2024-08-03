"""
Author: Jess Breda
Date: July 18, 2024
Description: single day plots that are general and can be used 
across protocols. Both for within days and across days.
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
from behav_viz.visualize.df_preperation import (
    rename_give_types,
    rename_curricula,
    compute_failed_fixation_rate_penalty_off,
)

######################################################################################
#########                        SINGLE DAY PLOTS                            #########
######################################################################################


######################################################################################
#########                        MULTI DAY PLOTS                             #########
######################################################################################


############################ COMBO SCATTER PLOTS ######################################
def plot_curriculum_and_give_types(
    trials_df, ax=None, title="", legend=False, rotate_x_labels=False
):
    """
    Make a scatter plot with both give type & curriculum info
    on the same plot.

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `give_type_imp` and `curriculum`
        with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str, (default = "")
        title of plot
    legend : bool (optional, default = True)
        whether to include the legend or not for give plot
    rotate_x_labels : bool (optional, default = False)
    """

    if ax is None:
        fig, ax = pu.make_fig()

    # Plot the give types
    plot_give_type_days(trials_df, ax=ax, legend=legend)

    # Plot the curriculum
    plot_curriculum(trials_df, ax=ax)

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(title=title, ylabel="", xlabel="")
    pu.set_legend(ax, legend)
    ax.grid(which="both")

    return None


############################ TRIALS & RUN TIME ######################################


def plot_run_time(days_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the run period for hours of the day over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `starttime_hrs` and
        `endtime_hrs` with dates as row index

    ax : matplotlib.axes.Axes, optional
        axes to plot on, if None, a new figure is created
    title : str, optional
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    if ax is None:
        fig, ax = pu.make_fig()
    ax.grid(alpha=0.5, zorder=1)

    if "endtime_hrs" not in days_df.columns:
        return None

    for _, row in days_df.iterrows():
        duration = row["endtime_hrs"] - row["starttime_hrs"]
        ax.bar(
            row["date"],
            duration,
            bottom=row["starttime_hrs"],
            width=0.8,
            align="center",
            color="azure",
            edgecolor="black",
            hatch="//",
        )
        ax.text(
            row["date"],
            row["starttime_hrs"] - 0.1,  # position the text slightly above the bar
            f"{duration:.2f}",
            ha="center",  # horizontal alignment
            va="bottom",  # vertical alignment
            fontsize=10,  # font size
            color="black",  # text color
        )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(ylabel="Time of Day (Hrs)", xlabel="", title=title)
    ax.set_ylim(days_df.starttime_hrs.min() - 1, days_df.endtime_hrs.max() + 1)
    ax.set_ylim(days_df.starttime_hrs.min() - 1, days_df.endtime_hrs.max() + 1)
    ax.invert_yaxis()

    return None


############################ CURRICULUM ######################################


def plot_curriculum(trials_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the curriculum information across days.

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `curriculum` with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str, (default = "")
        title of plot
    rotate_x_labels : bool (optional, default = False)
    """
    data = trials_df[["date", "curriculum"]].copy()
    data["curriculum"] = rename_curricula(data)

    if ax is None:
        fig, ax = pu.make_fig()

    sns.scatterplot(
        data=data,
        x="date",
        y="curriculum",
        color="black",
        ax=ax,
        s=250,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(title=title, ylabel="", xlabel="")
    ax.grid(which="both")

    return None


############################ GIVE TYPE ######################################


def plot_give_type_days(
    trials_df, ax=None, title="", legend=False, rotate_x_labels=False
):
    """
    Plot the give information across days.

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `give_type_imp` with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str, (default = "")
        title of plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
    """
    # make the names shorter for plotting
    data = trials_df[["date", "give_type_imp"]].copy()
    data["give_type_imp"] = rename_give_types(data)

    if ax is None:
        fig, ax = pu.make_fig()

    sns.scatterplot(
        data=data,
        x="date",
        y="give_type_imp",
        hue="give_type_imp",
        hue_order=["n", "l", "w", "w + l"],
        palette=[
            "green",
            "gold",
            "cyan",
            "cornflowerblue",
        ],
        ax=ax,
        s=250,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(title=title, ylabel="", xlabel="")
    pu.set_legend(ax, legend)
    ax.grid(which="both")

    return None


############################ RIG & TECH INFO ######################################


def plot_rig_tech(days_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the tech and rig id over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `rigid` and `tech` with
        dates as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not

    """
    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(data=days_df, x="date", y="rigid", marker="o", color="gray", ax=ax)
    sns.lineplot(data=days_df, x="date", y="tech", marker="o", color="purple", ax=ax)

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(ylabel="Tech || Rig", xlabel="", title=title)
    ax.grid()

    return None


def plot_rig_tech_foodpuck(days_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the tech and rig id over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `rigid` and `tech` with
        dates as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not

    """
    days_df["foodpuck_cat"] = days_df.foodpuck.apply(lambda x: "Y" if x > 0 else "N")

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(data=days_df, x="date", y="rigid", marker="o", color="gray", ax=ax)
    sns.lineplot(data=days_df, x="date", y="tech", marker="o", color="purple", ax=ax)
    sns.lineplot(
        data=days_df, x="date", y="foodpuck_cat", marker="o", color="teal", ax=ax
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(ylabel="Food Puck || Tech || Rig ", xlabel="", title=title)
    ax.grid()

    return None


############################ FAILED FIX (VIOLATIONS) ######################################


def plot_failed_fixation_rate_penalty_off(
    trials_df, only_trial_failures=False, ax=None, title="", rotate_x_labels=False
):
    """
    Plot the failed fixation rates by trial and by poke across days
    when the violation penalty is off

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date` and
        `n_settling_ins` with trials as row index
    only_trial_failures : bool (optional, default = False)
        whether to only plot the failed fixation rate by trial
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        fig, ax = pu.make_fig()

    failed_fix_df = compute_failed_fixation_rate_penalty_off(trials_df)

    if only_trial_failures:
        failed_fix_df.drop(
            failed_fix_df[failed_fix_df.type == "by_poke"].index, inplace=True
        )

    sns.lineplot(
        data=failed_fix_df,
        x="date",
        y="failure_rate",
        hue="type",
        hue_order=["by_trial", "by_poke"],
        palette=["salmon", "purple"],
        marker="o",
        ax=ax,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(ylabel="Failed Fixation Rate", xlabel="", title=title, ylim=(-0.1, 1.1))
    ax.grid()

    return None


def plot_failed_fixation_rate_penalty_on(
    trials_df, ax=None, title="", rotate_x_labels=False
):
    """
    Plot the failed fixation rates by trial and by poke across days
    when the violation penalty is on. In otherwords, this is just
    the violation rate.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date` and
        `n_settling_ins` with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=trials_df,
        x="date",
        y="violations",
        color="orangered",
        marker="o",
        ax=ax,
        errorbar=None,
        label="by_viol",
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(ylabel="Failed Fixation Rate", xlabel="", title=title, ylim=(-0.1, 1.1))
    ax.grid()

    return None


def plot_n_settling_ins_days(trials_df, ax=None, title="", rotate_x_labels=False):

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=trials_df,
        x="date",
        y="n_settling_ins",
        color="blue",
        marker="o",
        ax=ax,
    )
    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(ylabel="Avg N Settling Ins", xlabel="", title=title, ylim=(-0.1, None))
    ax.grid()

    return None
