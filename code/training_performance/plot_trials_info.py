"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting functions for looking at training 
performance within a day across trials
"""

import pandas as pd
import seaborn as sns
import plot_utils as pu
import numpy as np

# TODO move plots from `DMS_utils` to here


#### RESPONSE INFO ####
def plot_results(trials_df, ax, title=""):
    """
    plot trial result across a single day

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `result`, with
        trials as row index
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


def plot_result_summary(trials_df, ax, title=""):
    """
    plot trial result summary bar chart across a single day

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `result`, with
        trails as row index
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot

    """
    # get the proportion of each result
    res_summary = trials_df.result.value_counts(normalize=True).sort_index(
        ascending=True
    )

    # plot the bar chart
    res_summary.plot(kind="bar", color=pu.get_result_colors(trials_df.result), ax=ax)

    # aesthetics
    _ = ax.bar_label(ax.containers[0], fontsize=12, fmt="%.2f")
    ax.set(ylim=(0, 1), ylabel="performance", xlabel="", title=title)
    ax.set_xticklabels(pu.get_result_labels(trials_df.result), rotation=45)

    return None


#### SIDE INFO ####
def plot_correct_side(trials_df, ax, title=""):
    """
    plot correct side across a single day

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `sides`, with
        trials as row index and `sides` as `l` or `r`
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
        hue_order=pu.get_side_order(trials_df.sides),
        palette=pu.get_side_colors(trials_df.sides),
        legend=False,
    )
    _ = ax.set(ylabel="Side", xlabel="Trials", title=title)

    return None


def plot_side_bias_summary(trials_df, ax):
    """
    plot the side bias for a single day

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `sides`, with
        trials as row index and `sides` as `l` or `r`
    ax : matplotlib.axes
        axis to plot to
    """

    # get the hit rate for each side
    lr_hits = trials_df.groupby("sides").hits.mean().sort_index()

    # plot the bar chart
    lr_hits.plot(kind="bar", color=pu.get_side_colors(trials_df.sides), ax=ax)

    # calculate overall bias for plot title if there are both sides
    if len(lr_hits) == 2:
        bias = lr_hits["l"] - lr_hits["r"]
    else:
        bias = np.nan

    # aesthetics
    _ = ax.set(
        ylim=(0, 1), title=f"Bias: {np.round(bias, 2)}", xlabel="", ylabel="performance"
    )

    return None


def plot_side_count_summary(trials_df, ax):
    """
    plot the correct side count for a single day

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `result`, with
        trials as row index and `sides` as `l` or `r`
    ax : matplotlib.axes
        axis to plot to
    """
    # get the trial count for each side
    side_count = trials_df.sides.value_counts().sort_index()

    # plot the bar chart
    side_count.plot(kind="bar", color=pu.get_side_colors(trials_df.sides), ax=ax)

    # calculate overall ratio for plot title if there are both sides
    if len(side_count) == 2:
        lr_ratio = side_count["l"] / side_count["r"]
    else:
        lr_ratio = np.nan
    ax.set(
        title=f"L to R Ratio: {np.round(lr_ratio,2)}",
        xlabel="",
        ylabel="number of trials",
    )

    return None


### POKING INFO ###


def plot_npokes(trials_df, ax, title="", legend=True):
    """
    plot number of pokes (l,c,r) over trials for a single day

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `n_lpokes`,
        `n_cpokes`, `n_rpokes`, with trials as row index
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    legend : bool, (default = True)
        whether to include legend or not
    """
    # make long form df
    pokes_df = pd.melt(
        trials_df,
        id_vars=["trial"],
        value_vars=["n_lpokes", "n_cpokes", "n_rpokes"],
        var_name="poke_side",
        value_name="n_pokes",
    )

    sns.lineplot(
        data=pokes_df,
        x="trial",
        y="n_pokes",
        # note this could throw an error if no pokes happened in a port
        palette=pu.get_side_colors(pd.Series(["l", "c", "r"])),
        hue="poke_side",
        ax=ax,
    )

    # aesthetics
    _ = ax.set(ylabel="N pokes", xlabel="Trials", title=title)
    pu.set_legend(ax, legend)

    return None


def plot_npokes_summary(trials_df, ax, title=""):
    """
    plot summary of number of pokes per trial for a single day

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `n_lpokes`,
        `n_cpokes`, `n_rpokes`, with trials as row index
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    """

    # crete sides for color/labels
    sides = ["l", "c", "r"]

    # make long form df
    pokes_df = pd.melt(
        trials_df,
        id_vars=["trial"],
        value_vars=["n_lpokes", "n_cpokes", "n_rpokes"],
        var_name="poke_side",
        value_name="n_pokes",
    )

    # plot stripplot
    sns.stripplot(
        data=pokes_df,
        x="poke_side",
        y="n_pokes",
        palette=pu.get_side_colors(pd.Series(sides)),
        hue="poke_side",
        size=4,
        ax=ax,
        legend=False,
    )

    # plot boxplot
    sns.boxplot(
        data=pokes_df,
        x="poke_side",
        y="n_pokes",
        width=0.5,
        showfliers=False,
        linewidth=1,
        boxprops={"facecolor": "None"},
        ax=ax,
    )

    # aesthetics
    ax.set_xticklabels(sides)
    ax.set(ylabel="N pokes", xlabel="", title=title)

    return None


def plot_time_to_first_spoke(trials_df, ax, title="", legend=False):
    """
    plot time to first spoke across trials for a single day, this
    plot only shows the first spoke for each trial- not the first
    spoke for each side, for each trial.

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `min_time_to_spoke`,
        `first_spoke` with trials as row index
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    legend : bool, (default = False)
        whether to include legend or not
    """
    sns.scatterplot(
        data=trials_df,
        x="trial",
        y="min_time_to_spoke",
        ax=ax,
        hue="first_spoke",
        palette=pu.get_side_colors(trials_df.first_spoke),
    )

    # # mark
    # was_no_answer = df["result"].apply(lambda row: True if row == 6.0 else np.nan)
    # ax.scatter(x=trials_df.trial, y=was_no_answer, marker="s", color="black")

    # aesthetics
    upper_lim = trials_df["min_time_to_spoke"].quantile(0.99)  # exclude outliers
    ax.set_ylim(0.1, upper_lim)
    _ = ax.set(ylabel="1st Spoke Time [s]", xlabel="Trials", title=title)
    pu.set_legend(ax, legend)


def plot_first_spokes_summary_by_correct_side_and_location(
    trials_df, ax, title="", legend=False
):
    """
    plot first spoke summary by correct side for a day.
    time_to_first_spoke ~ correct_side + poke_location.
    Note this plot does show the first spoke for each side,
    for each trial. So on a left trial, if the mouse poked left first,
    then right, the first spoke for left will be plotted under the
    "correct" side in green and the first spoke for right will be
    plotted under the "incorrect" side, colored in red

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `first_spoke`,
        `result`, `sides` `first_lpoke`,  `first_rpoke` with
        trials as row index
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    legend : bool, (default = False)
        whether to include legend or not
    """

    # make long df for first l/r poke on each trial
    first_spoke_df = pd.melt(
        trials_df,
        id_vars=["trial", "first_spoke", "result", "sides"],
        value_vars=["first_lpoke", "first_rpoke"],
        var_name="poke_side",
        value_name="poke_time",
    )

    # determine if poke was on correct side
    first_spoke_df["correct_side"] = first_spoke_df.sides == first_spoke_df.first_spoke

    # rename for plotting
    first_spoke_df["poke_side"] = first_spoke_df.poke_side.str.replace(
        "first_", ""
    ).str.replace("poke", "")

    sns.stripplot(
        data=first_spoke_df,
        x="correct_side",
        y="poke_time",
        hue="poke_side",
        palette=pu.get_side_colors(first_spoke_df.poke_side),
        ax=ax,
        dodge=True,
        alpha=0.5,
        jitter=0.2,
    )

    sns.boxplot(
        data=first_spoke_df,
        x="correct_side",
        y="poke_time",
        hue="poke_side",
        showfliers=False,
        dodge=True,
        width=0.8,
        linewidth=1,
        boxprops={"facecolor": "None"},
        ax=ax,
    )

    # aesthetics
    upper_lim = first_spoke_df["poke_time"].quantile(0.99)  # exclude outliers
    ax.set_ylim(0.1, upper_lim)
    ax.set(ylabel="Spoke Time [s]", xlabel="Correct")
    pu.set_legend(ax, legend)


def plot_first_spoke_summary_by_location_and_result(
    trials_df, ax, title="", legend=False
):
    """
    plot first spoke summary given the location of the poke and
    the result of the trial. Note this only looks at the first
    spoke in a trial, not for each side, for each trial.

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `min_time_to_spoke`,
        `result`, `first_spoke` with trials as row index
    ax : matplotlib axis
        axis to plot on
    title : str, (default = "")
        title of plot
    legend : bool, (default = False)
        whether to include legend or not
    """
    sns.stripplot(
        data=trials_df,
        x="first_spoke",
        y="min_time_to_spoke",
        hue="result",
        hue_order=pu.get_result_order(trials_df.result),
        palette=pu.get_result_colors(trials_df.result),
        ax=ax,
        dodge=True,
        alpha=0.5,
        jitter=0.2,
    )
    sns.boxplot(
        data=trials_df,
        x="first_spoke",
        y="min_time_to_spoke",
        hue="result",
        showfliers=False,
        dodge=True,
        width=0.8,
        linewidth=1,
        boxprops={"facecolor": "None"},
        ax=ax,
    )

    # aesthetics
    upper_lim = trials_df["min_time_to_spoke"].quantile(0.99)  # exclude outliers
    ax.set_ylim(0.1, upper_lim)
    ax.set(title=title, xlabel="1st Spoke", ylabel="Time [s]")
    pu.set_legend(ax, legend)

    return None
