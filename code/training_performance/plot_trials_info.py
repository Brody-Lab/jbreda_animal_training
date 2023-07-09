"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting functions for looking at training 
performance within a day across trials
"""

import pandas as pd
import seaborn as sns
import plot_utils as pu

# TODO move plots from `DMS_utils` to here


#### RESPONSE INFO ####
def plot_results(trials_df, ax, title=""):
    """
    plot trial result across a single day

    params
    ------
    trials_df : DataFrame
        days dataframe with columns `trial`, `result`, with
        dates as row index
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    """
    sns.scatterplot(
        data=trials_df,
        x="trial",
        y="result",
        ax=ax,
        hue=trials_df["result"].astype("category"),
        hue_order=pu.get_result_order(trials_df.result),
        palette=pu.get_result_colors(trials_df.result),
        legend=False,
    )

    _ = ax.set(ylim=(0, 7), ylabel="Results", xlabel="Trials", title=title)

    return None


#### SIDE INFO ####
def plot_correct_side(trials_df, ax, title=""):
    """
    plot correct side across a single day

    params
    ------
    trials_df : DataFrame
        days dataframe with columns `trial`, `sides`, with
        dates as row index and `sides` as `l` or `r`
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    """
    sns.scatterplot(
        data=trials_df,
        x="trial",
        y="sides",
        ax=ax,
        hue=trials_df.sides,
        hue_order=pu.get_poke_order(trials_df.sides),
        palette=pu.get_poke_colors(trials_df.sides),
        legend=False,
    )
    _ = ax.set(ylabel="Side", xlabel="Trials", title=title)

    return None
