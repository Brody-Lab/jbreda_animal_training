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

######################################################################################
#########                        SINGLE DAY PLOTS                            #########
######################################################################################


######################################################################################
#########                        MULTI DAY PLOTS                             #########
######################################################################################


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


############################ CENTER POKING STATS ######################################
