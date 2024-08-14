"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting functions for looking at training 
performance within a day across trials
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
from behav_viz.ingest.create_days_df import create_days_df_from_dj  # for water plot


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

    _ = ax.set(ylim=(0, 7), ylabel="Results", xlabel="Trial", title=title)

    return None


def plot_result_summary(trials_df, ax, title=""):
    """
    plot trial result summary bar chart across a single day

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


def plot_performance_rates(trials_df, ax, title="", legend=True):
    """
    plot performance rates across a single day

    params
    ------
    trials_df: pd.DataFrame
        trials dataframe with columns: `trial`,`violation_rate`,
        `error_rate`, `hit_rate` with trials as row index
    ax: matplotlib.axes.Axes
        axes to plot to
    title : str, (default = "")
        title of plot
    legend : bool, (default = False)
        whether to include legend or not
    """
    # make long df
    perf_rates_df = pd.melt(
        trials_df,
        id_vars=["trial"],
        value_vars=["violation_rate", "error_rate", "hit_rate"],
        var_name="perf_type",
        value_name="rate",
    )
    sns.lineplot(
        data=perf_rates_df,
        x="trial",
        y="rate",
        hue="perf_type",
        palette=["orangered", "maroon", "darkgreen"],
        ax=ax,
    )

    # aesthetics
    _ = ax.set(ylabel="Rate", xlabel="Trial", title=title, ylim=(0, 1))
    ax.grid(alpha=0.5)
    pu.set_legend(ax, legend=legend)

    return None


def plot_stim_grid_performance(trials_df, ax, mode, title=""):
    """
    plot performance (viol or hit) in sa vs sb grid
    with rule boundary marked

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `sa` `sb`, `hits`
        and `violations`
        with trials as row index
    ax: matplotlib.axes.Axes
        axes to plot to, if None, create new axes
    mode : str
        mode to plot, either "violations" or "hits"
    """
    stim_table = (
        trials_df.groupby(["sa", "sb"])
        .agg(perf_rate=(mode, "mean"), perf_count=(mode, "size"))
        .reset_index()
    )

    # plot each sa,sb pair with rate as color
    scatter = ax.scatter(
        stim_table.sa,
        stim_table.sb,
        c=stim_table.perf_rate,
        cmap="flare",
        vmin=0,
        vmax=1,
        marker=",",
        s=100,
    )

    # Add a colorbar to the plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"{mode[:-1]} rate")

    # add labels to each point
    for i, txt in enumerate(stim_table.perf_rate):
        ax.text(
            stim_table.sa[i] - 1,
            stim_table.sb[i] + 0.5,
            f"{round(txt, 2)} [{stim_table.perf_count[i]}]",
            fontsize=8,
        )

    ## Aesthetics
    # match/non-match boundary line
    ax.axline((0, 1), slope=1, color="lightgray", linestyle="--")
    ax.axline((1, 0), slope=1, color="lightgray", linestyle="--")

    # Range& aesthetics
    # sp_min, sp_max = np.min(stim_table.sa), np.max(stim_table.sa)
    # stim_range = [sp_min, sp_max]
    x_lim = [0, 15]
    y_lim = [0, 15]

    _ = ax.set(
        xlabel="Sa",
        ylabel="Sb",
        xlim=x_lim,
        ylim=y_lim,
        xticks=[3, 12],
        yticks=[3, 12],
        title=title,
    )

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
    _ = ax.set(ylabel="Side", xlabel="Trial", title=title)

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


def plot_antibias_r_probs(trials_df, ax=None, title=""):
    """
    plot the probability of a right trial across trials
    note: this is only in effect if antibias beta > 0
    for animals settings file

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `trial` `ab_r_prob`,
        with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    title : str, (default = "")
        title of plot
    """
    ax = plt.gca() if ax is None else ax

    sns.lineplot(
        data=trials_df, x="trial", y="ab_r_prob", marker="o", color="firebrick", ax=ax
    )
    ax.axhline(0.5, ls="--", color="black")

    # aesthetics
    _ = ax.set(ylim=(0, 1), xlabel="trial", ylabel="Antibias Prob R ", title=title)

    return None


### STAGE INFO ###


def plot_stage_info(trials_df, ax):
    """
    plotting a bar indicating stage at top of plot

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with colums `sides` and `stage` with
        trials as row idex
    ax: matplotlib.axes.Axes
        axes to plot to

    notes
    ----
    NOTE this code assumes you are using this with
    plot_correct_side() and the y axis is l and/or r
    """

    # the ax.text get's really weird when there is only 1 category
    # for the y axis, so we need to adjust the y position manually
    if trials_df.sides.nunique() == 1:
        y_position = 0
    else:
        y_position = 0.5

    gray_palette = sns.color_palette("gray")

    # iterate over trials in each stage, make the hbar + plot
    # need to make sure this doesn't include crash trials
    # which are marked as 5 and make stage = nan
    for i, s in enumerate(trials_df.query("result != 5").stage.unique()):
        # calculate trial start and stop numbers for a given stage
        bounds = trials_df.query("stage == @s").agg({"trial": ["min", "max"]})
        ax.axvspan(
            xmin=bounds.iloc[0, 0],
            xmax=bounds.iloc[1, 0] + 1,
            ymin=0.25,
            ymax=0.75,  # relative to plot (0,1)
            alpha=0.3,
            color=gray_palette[int(i - 1)],
        )
        # add text label
        ax.text(
            x=bounds.mean().values[0],
            y=y_position,
            s=f"stage {int(s)}",
            ha="center",
            va="center",
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
    _ = ax.set(ylabel="N pokes", xlabel="Trial", title=title)
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
    ax.set_xticks([0, 1, 2])  # Set the positions of the ticks
    ax.set_xticklabels(sides)  # Set the tick labels
    ax.set(ylabel="N pokes", xlabel="", title=title)

    return None


#### CENTER POKING ####


def plot_n_settling_ins(trials_df, ax):
    """
    plot histogram of the number of settling ins
    per trial

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `n_settling_ins`
        with trials as row index
    ax : matplotlib.axes
        axis to plot to
    """

    if trials_df.n_settling_ins.nunique():
        binwidth = None
    else:
        binwidth = 1

    sns.histplot(trials_df.n_settling_ins, binwidth=binwidth, ax=ax)
    ax.axvline(trials_df.n_settling_ins.mean(), color="k", linestyle="--", lw=3)

    ax.set(
        xlabel="N failed / trial",
        title=f"Mean N Settling: {trials_df.n_settling_ins.mean():.2f}",
    )

    return None


def plot_ncpokes_over_trials(trials_df, ax, title=""):
    """
    TODO
    """
    sns.lineplot(x="trial", y="n_settling_ins", data=trials_df, ax=ax)

    ax.set(xlabel="Trial", ylabel="N Cpokes", title=title)
    ax.grid()

    return None


#### SIDE POKING ####
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
    ax.set_ylim(top=upper_lim)
    _ = ax.set(ylabel="1st Spoke Time [s]", xlabel="Trial", title=title)
    pu.set_legend(ax, legend)

    return None


def plot_first_spokes_summary_by_correct_side_and_loc(
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
        order=[True, False],  # put correct on left
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
        order=[True, False],  # put correct on left
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
    ax.set(ylabel="Spoke Time [s]", xlabel="Correct", title=title)
    pu.set_legend(ax, legend)

    return None


def plot_first_spoke_summary_by_loc_and_result(trials_df, ax, title="", legend=False):
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

    if (trials_df["min_time_to_spoke"]).isna().all():
        return None

    sns.stripplot(
        data=trials_df.query("first_spoke != 'n'"),
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
        data=trials_df.query("first_spoke != 'n'"),
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


#### TRIAL LENGTH ####
def plot_trial_dur(trials_df, ax, title="", legend=True):
    """
    plot trial duration and iti across a single day

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `trial`, `trial_dur`,
        and `inter_trial_dur` with trials as row index
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    legend : bool, (default = False)
        whether to include legend or not
    """
    # make long df
    trial_dur_df = pd.melt(
        trials_df,
        id_vars=["trial"],
        value_vars=["trial_dur", "inter_trial_dur"],
        var_name="duration_type",
        value_name="duration",
    )
    sns.lineplot(
        data=trial_dur_df,
        x="trial",
        y="duration",
        palette=["cornflowerblue", "gray"],
        hue="duration_type",
        ax=ax,
    )
    ax.axhline(
        trials_df.trial_dur.mean(), color="cornflowerblue", linestyle="--", zorder=1
    )

    active_time = trials_df.trial_dur - trials_df.inter_trial_dur

    # aesthetics
    _ = ax.set(
        ylabel="Duration [s]",
        xlabel="Trial",
        title=f"Avg SMA ITI: {trials_df.inter_trial_dur.mean():.2f}, Avg Animal ITI: {active_time.mean():.2f}",
    )
    pu.set_legend(ax, legend=legend)

    return None


def plot_active_trial_dur_summary(trials_df, ax):
    """
    plot histogram of active trial duration, ie how long was
    the trial when the iti duration is removed

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial_dur` and `inter_trial_dur`
        with trials as row index
    ax : matplotlib.axes
        axis to plot to
    """

    # remove inter trial duration time- animal is just waiting
    active_time = trials_df.trial_dur - trials_df.inter_trial_dur

    # plot
    if active_time.max() > 100:
        log_scale = True
        units = "log(s)"
    else:
        log_scale = False
        units = "s"

    sns.histplot(active_time, bins=30, element="step", log_scale=log_scale, ax=ax)

    # aesthetics
    ax.set(
        xlabel=f"Active Trial Dur [{units}]",
        ylabel="Count",
        title=f"Avg Active Dur: {active_time.mean():.2f} s",
    )

    return None


#### GIVE ####
def plot_give_info(trials_df, ax, legend=False):
    """
    plot give info for each trial. will plot the give type that
    was set by the GUI and the give type that was implemented by
    the underlying code (e.g. if fraction of give trials is not
    100%, then the give type set by the GUI will not match the
    type implemented every trial)

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial`, `give_type_imp`, and
        `give_type_set` with trials as row index
    ax : matplotlib.axes.Axes
        axis to plot to
    """

    # make the names shorter for plotting
    data = trials_df[["trial", "give_type_imp", "give_type_set"]].copy()
    mapping = {"water_and_light": "w + l", "water": "w", "light": "l", "none": "n"}
    data.give_type_imp = data.give_type_imp.replace(mapping)

    sns.scatterplot(
        data=data,
        x="trial",
        y="give_type_imp",
        hue="give_type_imp",
        palette=pu.get_give_colors(data["give_type_imp"]),
        hue_order=pu.get_give_order(data["give_type_imp"]),
        ax=ax,
    )

    _ = ax.set(title="Give Type Implemented", xlabel="Trial", ylabel="")

    pu.set_legend(ax, legend)

    return None


def plot_give_count_summary(trials_df, ax, title=""):
    """
    plot the count of give types implemented and color by
    what was actually set in the GUI

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `give_type_imp` and `give_type_set`
        with trials as row index
    ax : matplotlib.axes
        axis to plot to

    TODO can update with give colors eventually
    """
    # summarize counts
    count_data = (
        trials_df.groupby(["give_type_imp", "give_type_set"])
        .size()
        .reset_index(name="count")
    )
    # make the names shorter for plotting
    mapping = {"water_and_light": "w + l", "water": "w", "light": "l", "none": "n"}
    count_data.replace(
        {"give_type_imp": mapping, "give_type_set": mapping}, inplace=True
    )

    # plot
    sns.barplot(
        data=count_data,
        x="give_type_imp",
        y="count",
        hue="give_type_set",
        palette=pu.get_give_colors(count_data["give_type_imp"]),
        hue_order=pu.get_give_order(count_data["give_type_imp"]),
        ax=ax,
    )
    # aesthetics
    ax.set(
        title="Give Stats" if title == "" else title,
        xlabel="Implemented",
        ylabel="Count",
    )
    ax.legend(title="Set")

    return None


def plot_result_by_give(trials_df, ax=None, title="", legend=False):
    """
    plot result by give type implemented

    params
    -----
    trials_df: pd.DataFrame
        trials dataframe with columns: `give_type_imp`,
        `result` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    title : str, (default = "")
        title of plot
    legend : bool (default = False)
        whether to include legend or not
    """
    ax = plt.gca() if ax is None else ax

    result_by_give = (
        trials_df.groupby("give_type_imp").result.value_counts().reset_index()
    )

    sns.barplot(
        data=result_by_give,
        x="give_type_imp",
        y="count",
        hue="result",
        hue_order=pu.get_result_order(trials_df.result),
        palette=pu.get_result_colors(trials_df.result),
        ax=ax,
    )

    # aesthetics
    ax.set(xlabel="Give Type", ylabel="Count", title=title)
    pu.set_legend(ax, legend=legend)

    return None


def plot_hit_rate_by_give(trials_df, ax=None, title=""):
    """
    plot hit rate by give type implemented

    params
    -----
    trials_df: pd.DataFrame
        trials dataframe with columns: `give_type_imp`,
        `hits` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    title : str, (default = "")
        title of plot
    """
    ax = plt.gca() if ax is None else ax

    hit_rate_by_give = trials_df.groupby("give_type_imp").hits.mean().reset_index()
    sns.barplot(
        x="give_type_imp",
        hue="give_type_imp",
        y="hits",
        data=hit_rate_by_give,
        ax=ax,
        order=pu.get_give_order(hit_rate_by_give.give_type_imp),
        palette=pu.get_give_colors(hit_rate_by_give.give_type_imp),
    )

    # aesthetics
    ax.set(xlabel="Give Type", ylabel="Hit Rate", title=title)

    return None


#### WATER ####
def plot_watering_amounts(
    trials_df: pd.DataFrame, ax=None, title: str = "", legend: bool = False
) -> None:
    """
    plot the water drunk in the rig and pub for a day with
    restriction volume indicated. note the trials_df is only
    being used to find the animal_id and date to fetch the
    mass/water information with

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `animal_id` and `date`
        with only one unique value in each
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str, (default = "")
        title of plot
    legend : bool, (default = True)
        whether to include legend or not
    """
    assert (
        len(trials_df.animal_id.unique()) == 1 and len(trials_df.date.unique()) == 1
    ), print("trials_df must have only one unique animal_id and date")

    animal_id = trials_df.animal_id.iloc[0]
    date = str(trials_df.date.iloc[0])
    df = create_days_df_from_dj(animal_ids=[animal_id], date_min=date, date_max=date)

    if ax is None:
        fig, ax = pu.make_fig("s")

    bar_plot_cols = ["date", "rig_volume", "pub_volume"]
    df[bar_plot_cols].set_index("date").plot(
        kind="bar", stacked=True, color=["blue", "cyan"], ax=ax
    )

    volume_target = df["volume_target"].iloc[0]
    ax.axhline(y=volume_target, xmin=0.2, xmax=0.8, color="black")
    # label the amounts
    ax.text(x=-0.45, y=volume_target, s=str(volume_target), fontsize=12)
    for cont in ax.containers:
        _ = ax.bar_label(
            cont, fontsize=12, fmt="%.2f", label_type="center", color="white"
        )

    ax.set_xticks([])
    _ = ax.set(xlabel="", ylabel="volume (mL)", title=title)
    pu.set_legend(ax, legend=legend)

    return None


### INTRODUCING STIMULI ###
def plot_stimulus_and_delay_durations(trials_df, ax, title="", legend=True):
    """
    plot the stimulus and delay durations for each trial,
    specifically for use in stages when the two are growing/shrinking
    in relationship to each other.

    params
    ------
    trials_df : DataFrame
        trials dataframe with columns `trial` `stimulus_dur`
        `delay_dur` with trials as row index
    ax : matplotlib.axes.Axes
        axis to plot to
    title : str, (default = "Stim & Delay Periods Durations")
        title of plot
    legend : bool, (default = False)
        whether to include legend or not
    """

    # make long df
    stim_del_dur_df = trials_df.melt(
        id_vars=["trial"],
        value_vars=["stimulus_dur", "delay_dur"],
        var_name="period",
        value_name="duration",
    )
    # plot
    sns.lineplot(
        x="trial",
        y="duration",
        hue="period",
        hue_order=["stimulus_dur", "delay_dur"],
        palette=[pu.TRIAL_PERIOD_MAP["s_a"], pu.TRIAL_PERIOD_MAP["delay"]],
        data=stim_del_dur_df,
        ax=ax,
    )
    # aesthetics
    ax.set(
        title="Stim & Delay Periods Durations" if title == "" else title,
        xlabel="Trial",
        ylabel="Duration [s]",
    )

    return None


### VIOLATIONS ###


def create_viol_period_df(trials_df):
    """
    create a dataframe of violation only trials and
    determine which trial period the violation occurred in
    for plotting purposes

    params
    ------
    trials_df: pandas.DataFrame
        trials dataframe with columns listed within this
        fx and trials as row index

    returns
    -------
    viols_df: pandas.DataFrame
        dataframe of only violation trials with period
        end times and a volume ("violation_period") indicating
        which period violation occurred in

    NOTE: the column names have been adjusted for plotting to
    remove the _dur from the end of the column name and add sa/sb
    so they do not match trials_df column names
    """

    # grab columns of interest & and only violations
    columns = [
        "settling_in_dur",
        "adj_pre_dur",
        "stimulus_dur",
        "delay_dur",
        "post_dur",
        "cpoke_dur",
        "pre_go_dur",
        "trial",
        "violations",
    ]
    viols_df = trials_df[columns].query("violations == 1")

    # remove _dur from column names & change stimulus to s_a or s_b
    viols_df.insert(5, "s_b", viols_df["stimulus_dur"])
    viols_df.rename(columns={"stimulus_dur": "s_a"}, inplace=True)
    viols_df.columns = viols_df.columns.str.replace("_dur", "")

    # TODO- these should not be nans!
    viols_df.dropna(subset=["cpoke"], inplace=True)

    # calculate the period times in series and add to df as period_end
    # currently only have the duration of the periods, but we know
    # their order and the end of the trial (pre_go_dur).
    periods = [
        "settling_in",
        "adj_pre",
        "s_a",
        "delay",
        "s_b",
        "post",
    ]
    elapsed_time = 0
    for period in periods:
        viols_df[period + "_end"] = elapsed_time + viols_df[period]
        elapsed_time = viols_df[period + "_end"].values

    def _determine_period(row):
        """
        quick function to determine which period the violation
        occurred in to be used with df.apply() on rows
        """
        for period in periods:
            if row["cpoke"] <= row[period + "_end"]:
                return period
        # a violation shouldn't happen after go, but there is a bug
        # in the code somewhere that i have yet to find, so we will
        # just mark it as go and color it white (August 2023)
        return "go"

    viols_df["violation_period"] = viols_df.apply(_determine_period, axis=1)

    return viols_df


def plot_violations_by_period(trials_df, ax, title="", legend=False):
    """
    plot a histogram of violations over time relative to the
    go cue & color by the period they occurred in

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with period columns [`settling_in_dur`,
        `adj_pre_dur`, `stimulus_dur`, `delay_dur`, `post_dur`,
        `cpoke_dur`], poking information [`cpoke_dur`], and trial
        information [`trial`, `violations`] with trials as
        row index.
    ax : matplotlib.axes.Axes
        axis to plot to
    title : str, (default = "Violation Histogram by Period")
        title of plot
    legend : bool, (default = False)
        whether to include legend or not
    """

    # create dataframe that marks the violation period
    viols_df = create_viol_period_df(trials_df)

    # make timing relative to go cue for alignment
    viols_df["cpoke"] = viols_df.cpoke - viols_df.pre_go

    hue_order = pu.get_period_order(viols_df.violation_period)
    palette = pu.get_period_colors(viols_df.violation_period)
    # plot
    _ = sns.histplot(
        data=viols_df,
        x="cpoke",
        hue="violation_period",
        hue_order=hue_order,
        palette=palette,
        binwidth=0.05,
        element="step",
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=3)
    # aesthetics
    _ = ax.set(
        title="Violation Histogram by Period" if title == "" else title,
        xlabel="Cpoke Dur Relative to Go [s]",
        ylabel="Count",
    )

    ax.get_legend().set_title("")
    # if not legend:  # weird bug with hist plot, hack around
    #     pu.set_legend(ax, legend)

    return None


### PRO ANTI


def plot_pro_anti_cumulative_perf_rates(trials_df, ax):
    """
    Plot cumulative performance rates for pro and anti trials
    across trials

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `trial`, `pro_hit_rate`,
        `anti_hit_rate`, `pro_stim_set`, `anti_stim_set` with
        trials as row index
    ax : matplotlib.axes.Axes
        axis to plot to
    """
    perf_rates_df = pd.melt(
        trials_df,
        id_vars=["trial"],
        value_vars=["pro_hit_rate", "anti_hit_rate"],
        var_name="perf_type",
        value_name="rate",
    )
    sns.lineplot(
        data=perf_rates_df,
        x="trial",
        y="rate",
        hue="perf_type",
        hue_order=["pro_hit_rate", "anti_hit_rate"],
        palette="husl",
        ax=ax,
        marker=".",
    )

    block_switch = trials_df["n_blocks"].diff().fillna(0).abs() > 0
    for trial in trials_df[block_switch].trial:
        ax.axvline(x=trial, color="black")

    ax.grid(alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--")

    _ = ax.set(
        xlabel="Trial",
        ylabel="Rate",
        title=f"Pro: {trials_df.pro_stim_set.unique()[0]}, Anti: {trials_df.anti_stim_set.unique()[0]}",
        ylim=(0, 1),
    )

    ax.grid(alpha=0.5)
    ax.legend(loc="lower left")

    return None


def rolling_avg(group, window_size, column="hits"):
    """
    helper function for rolling window to be applied to a group
    """
    group[f"{column}_rolling_avg_{window_size}"] = (
        group[column].rolling(window=window_size, min_periods=1).mean()
    )
    return group


def plot_rolling_hit_rate_by_pro_anti(trials_df, ax=None):
    """
    plot rolling hit rate by pro or anti

    params
    -----
    trials_df: pd.DataFrame
        trials dataframe with columns: `pro_anti_block_type`,
        `hits` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    """
    if ax is None:
        _, ax = pu.make_fig()

    window_size = min(int(trials_df.block_size.min()), 20)

    data = (
        trials_df.groupby("pro_anti_block_type")
        .apply(rolling_avg, window_size=window_size)
        .reset_index(drop=True)
    )

    sns.lineplot(
        data=data,
        x="trial",
        y=f"hits_rolling_avg_{window_size}",
        hue="pro_anti_block_type",
        hue_order=["pro", "anti"],
        palette="husl",
        ax=ax,
        marker=".",
    )

    block_switch = trials_df["n_blocks"].diff().fillna(0).abs() > 0
    for trial in trials_df[block_switch].trial:
        ax.axvline(x=trial, color="black")

    ax.grid(alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--")

    _ = ax.set(
        xlabel="Trial",
        ylabel="Performance Rate",
        title=f"Pro: {trials_df.pro_stim_set.unique()[0]}, Anti: {trials_df.anti_stim_set.unique()[0]}, Window Size: {window_size}",
        ylim=(0, 1),
    )

    ax.grid(alpha=0.5)
    ax.legend(loc="lower left")


def plot_pro_anti_count_summary(trials_df, ax=None):
    """
    plot the count of pro or anti trials for a given day

    params
    ------
    trials_df: pd.DataFrame
        trials dataframe with columns: `pro_anti_block_type`,
        `hits` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    """
    if ax is None:
        _, ax = pu.make_fig("s")

    count_data = (
        trials_df.groupby("pro_anti_block_type").size().reset_index(name="count")
    )

    sns.barplot(
        data=count_data,
        x="pro_anti_block_type",
        y="count",
        dodge=False,
        hue="pro_anti_block_type",
        palette="husl",
        order=["pro", "anti"],
        ax=ax,
    )

    _ = ax.set(
        xlabel="",
        ylabel="N Trials",
        title="Pro-Anti Trial Counts",
    )

    return None


def plot_hit_rate_by_pro_anti(trials_df, ax=None):
    """
    plot hit rate by pro or anti

    params
    -----
    trials_df: pd.DataFrame
        trials dataframe with columns: `pro_anti_block_type`,
        `hits` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    """
    if ax is None:
        _, ax = pu.make_fig("s")

    sns.barplot(
        data=trials_df.dropna(subset=["hits"]),  # need to drop nan for barplot
        x="pro_anti_block_type",
        hue="pro_anti_block_type",
        y="hits",
        palette="husl",
        order=["pro", "anti"],
        ax=ax,
    )
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)

    _ = ax.set(xlabel="", ylabel="Hit Rate", title="Pro-Anti Hit Rate", ylim=(0, 1))

    return None


def plot_rolling_hit_rate_by_stim(trials_df, ax=None):
    """
    plot rolling hit rate by stim pair

    params
    -----
    trials_df: pd.DataFrame
        trials dataframe with columns: `sound_pair`,
        `hits` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    """
    if ax is None:
        _, ax = pu.make_fig()

    window_size = min(int(trials_df.block_size.min()), 20)

    data = (
        trials_df.groupby("sound_pair")
        .apply(rolling_avg, window_size=window_size)
        .reset_index(drop=True)
    )

    sns.lineplot(
        data=data,
        x="trial",
        y=f"hits_rolling_avg_{window_size}",
        hue="sound_pair",
        palette=pu.create_palette_given_sounds(data),
        ax=ax,
        marker=".",
    )

    block_switch = trials_df["n_blocks"].diff().fillna(0).abs() > 0
    for trial in trials_df[block_switch].trial:
        ax.axvline(x=trial, color="black")

    ax.grid(alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--")

    _ = ax.set(
        xlabel="Trial",
        ylabel="Performance Rate",
        title=f"Pro: {trials_df.pro_stim_set.unique()[0]}, Anti: {trials_df.anti_stim_set.unique()[0]}, Window Size: {window_size}",
        ylim=(-0.1, 1.1),
    )

    ax.grid(alpha=0.5)
    ax.legend().remove()

    return None


### PRO & ANTI + Give Delay ###


def calc_anti_give_use_rate(trials_df):
    """
    Calculate the cumulative give_use_rate for anti trials in a trials_df

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with 'give_use' (bool) and `pro_anti_block_type`
        (str) columns with trials as row index

    returns
    -------
    anti_trials_df : pd.DataFrame
        trials dataframe with only anti trials and a new column
        'give_use_rate' which is the cumulative give_use rate
        for anti trials (since pro trials don't have giveß)
    """

    # Filter the DataFrame for rows where 'pro_anti_block_type' is 'anti'
    anti_trials_df = trials_df.query(
        'pro_anti_block_type == "anti" and violations != 1'
    ).copy()

    # Calculate cumulative sum of 'give_use' for 'anti' trials
    anti_trials_df["give_use_rate"] = anti_trials_df["give_use"].astype(
        int
    ).cumsum() / range(1, len(anti_trials_df) + 1)

    return anti_trials_df


def plot_anti_give_del_metrics(trials_df, ax=None, title="", legend=True):
    """
    Plot a series of related metric for anti trials specific to when
    the pre-give delay is growing for an animal on anti trials that
    are light guided.

    Specific metrics: give delay duration, anti hit rate, give delivery rate

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns: `pro_anti_block_type`,
        `give_delay_dur`, `anti_hit_rate`, `give_use_rate`
        with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    title : str, (default = "")
        title of plot
    legend : bool (default = True)
        whether to include legend or not
    """

    if ax is None:
        fig, ax = pu.make_fig()

    # Create df of only anti trials and calculate cumulative give_use_rate
    anti_trials_df = calc_anti_give_use_rate(trials_df)
    window_size = max(int(anti_trials_df.block_size.min()), 10)
    anti_trials_df = (
        anti_trials_df.query("violations != 1")
        .groupby("pro_anti_block_type")
        .apply(rolling_avg, window_size=window_size)
        .reset_index(drop=True)
    )
    anti_trials_df = (
        anti_trials_df.query("violations != 1")
        .groupby("pro_anti_block_type")
        .apply(rolling_avg, window_size=window_size, column="give_use")
        .reset_index(drop=True)
    )

    sns.lineplot(
        data=anti_trials_df,
        x="trial",
        y="give_delay_dur",
        marker="o",
        label="Anti Give Delay Dur",
        color="gray",
        ax=ax,
    )

    ax2 = ax.twinx()
    sns.lineplot(
        data=anti_trials_df,
        x="trial",
        y=f"hits_rolling_avg_{window_size}",
        marker="o",
        color=sns.color_palette("husl", 2)[1],
        label="Anti Rolling Hit",
        ax=ax2,
    )

    sns.lineplot(
        data=anti_trials_df,
        x="trial",
        y=f"give_use_rolling_avg_{window_size}",
        marker="o",
        color="gold",
        label="Anti Rolling Give Deliv.",
        ax=ax2,
    )

    ax2.axhline(
        anti_trials_df.give_del_adagrow_alpha_minus.min(), linestyle="-", color="k"
    )

    ax.set(
        ylabel="Give Delay [s]",
        xlabel="Trial",
        title=title + f"window size: {window_size}",
        xlim=(-10, None),
    )
    ax2.set(ylabel="Proportion", ylim=(0, 1))
    ax2.grid()

    return None


def plot_anti_hit_rate_by_give_use(trials_df, ax=None, title="", legend=True):
    """
    For anti trials, plot the hit rate for trials where give was used
    vs. trials where give was not used.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns: `pro_anti_block_type`,
        `give_use_rate` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    title : str, (default = "")
        title of plot
    legend : bool (default = True)
        whether to include legend or not

    """

    if ax is None:
        fig, ax = pu.make_fig("s")

    plot_data = trials_df.dropna(subset=["hits"]).query("pro_anti_block_type=='anti'")
    unique_hues = plot_data["give_use"].unique()
    hue_order = (
        [True, False] if all(x in unique_hues for x in [True, False]) else unique_hues
    )

    barplot = sns.barplot(
        data=plot_data,
        x="pro_anti_block_type",
        y="hits",
        hue="give_use",
        hue_order=hue_order,
        palette=["gold", "green"] if len(hue_order) > 1 else ["gold"],
        ax=ax,
    )

    # display counts of each give use type above each bar
    counts = (
        trials_df.dropna(subset=["hits"])
        .query("pro_anti_block_type=='anti'")
        .groupby("give_use")
        .size()
        .sort_index(ascending=False)  # True, False
    )

    for i, bar in enumerate(barplot.patches):
        # Get the count for this category
        count = counts[i]

        # Set the text above each bar
        ax.text(
            (bar.get_x() + bar.get_width() - 0.1) / 2,
            bar.get_height(),
            count,
            ha="right",
            va="bottom",
        )

    # aesthetics
    ax.set(ylabel="Hit Rate", xlabel="", title=title)
    if not legend:
        ax.get_legend().remove()

    return None


def plot_anti_give_use_counts(trials_df, ax=None, title="", legend=True):
    """
    For anti trials, plot the count of trials give was used
    vs. trials where give was not used.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns: `pro_anti_block_type`,
        `give_use_rate` with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    title : str, (default = "")
        title of plot
    legend : bool (default = True)
        whether to include legend or not
    """
    count_data = (
        trials_df.query("pro_anti_block_type == 'anti'")
        .groupby(["pro_anti_block_type", "give_use"])
        .size()
        .reset_index(name="count")
    )

    unique_hues = count_data["give_use"].unique()
    hue_order = (
        [True, False] if all(x in unique_hues for x in [True, False]) else unique_hues
    )

    if ax is None:
        fig, ax = pu.make_fig("s")

    sns.barplot(
        data=count_data,
        x="pro_anti_block_type",
        y="count",
        hue="give_use",
        hue_order=hue_order,
        palette=["gold", "green"],
        ax=ax,
    )

    # aesthetics
    ax.set(ylabel="Count", xlabel="", title=title)
    if not legend:
        ax.get_legend().remove()


def plot_anti_hit_counts_by_give_use(trials_df, ax=None, title="", legend=True):
    """
    For anti trials, plot the count of hits conditioned on
    where give was used vs. trials where give was not used.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns: `pro_anti_block_type`,
        `give_use_rate`, `hits`, with trials as row index
    ax: matplotlib.axes.Axes (default = None)
        axes to plot to, if None, create new axes
    title : str, (default = "")
        title of plot
    legend : bool (default = True)
        whether to include legend or not
    """

    count_data = (
        trials_df.query('pro_anti_block_type == "anti"')
        .groupby(["hits", "give_use"])
        .size()
        .reset_index(name="count")
    )

    if ax is None:
        fig, ax = pu.make_fig("s")

    sns.barplot(
        data=count_data,
        x="hits",
        y="count",
        hue="give_use",
        order=[True, False],
        hue_order=[True, False],
        palette=["gold", "green"],
        ax=ax,
    )

    ax.set(ylabel="Count", xlabel="Correct", title=title)

    if legend:
        ax.legend(title="give delivery", loc="upper left", frameon=False)
    else:
        ax.get_legend().remove()

    return None
