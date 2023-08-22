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
from IPython.display import clear_output

from create_days_df import fetch_and_format_single_day_water

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
    for i, s in enumerate(trials_df.stage.unique()):
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
    ax.set_xticklabels(sides)
    ax.set(ylabel="N pokes", xlabel="", title=title)

    return None


#### CENTER POKING ####


def plot_n_failed_cpokes(trials_df, ax):
    """
    plot histogram of the number of failed cpokes per trial
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

    upper_lim = trials_df["n_settling_ins"].quantile(0.99)  # exclude outliers
    ax.set_ylim(top=upper_lim)
    ax.set(
        xlabel="N failed / trial",
        title=f"Mean N failed cpoke: {trials_df.n_settling_ins.mean():.2f}",
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


def plot_cpoke_distributions(trials_df, ax, mode="settling_in", legend=False):
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
        ax.axvline(data.violated_trials.mean(), color=pal[1], lw=3)
        ax.axvline(data.valid_trials.mean(), color=pal[2], lw=3)

        multi_cpokes = np.sum(trials_df.n_settling_ins > 1) / len(trials_df)
        viol_rate = np.sum(trials_df.violations) / len(trials_df)

        ax.set(
            xlabel="Cpoke Dur Relative to Go [s]",
            title=f"Mutli-Settling {multi_cpokes:.2f}, Viol Rate: {viol_rate:.2f},  Avg Cpoke Dur: {avg_cpoke_dur:.2f}",
        )

    return None


def plot_cpokes_over_trials(trials_df, ax, mode="settling_in", title=""):
    """
    TODO
    """
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

    # aesthetics
    _ = ax.set(
        ylabel="Duration [s]",
        xlabel="Trial",
        title=f"Avg ITI: {trials_df.inter_trial_dur.mean():.2f}",
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
def plot_give_info(trials_df, ax, legend=True):
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

    sns.scatterplot(data=data, x="trial", y="give_type_imp", hue="give_type_set", ax=ax)

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
        data=count_data, x="give_type_imp", y="count", hue="give_type_set", ax=ax
    )
    # aesthetics
    ax.set(
        title="Give Stats" if title == "" else title,
        xlabel="Implemented",
        ylabel="Count",
    )
    ax.legend(title="Set")

    return None


#### WATER ####
# TODO update to take into account mouse/rate new code for the
# TODO threshold bar
def plot_watering_amounts(trials_df, ax, title="", legend=False):
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
    date = trials_df.date.iloc[0]

    # this makes a call to the create_days functions since they were
    # already well written to query the ratinfo database
    df, volume_target = fetch_and_format_single_day_water(animal_id, date)

    # plot stacked bar
    df.set_index("date").plot(kind="bar", stacked=True, color=["blue", "cyan"], ax=ax)
    # plot the target threshold
    ax.axhline(y=volume_target, xmin=0.2, xmax=0.8, color="black")
    # label the amounts
    ax.text(x=-0.45, y=volume_target, s=str(volume_target), fontsize=12)
    for cont in ax.containers:
        _ = ax.bar_label(
            cont, fontsize=12, fmt="%.2f", label_type="center", color="white"
        )

    # aesthetics
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
    # from IPython.display import clear_output

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
    clear_output(wait=True)
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
