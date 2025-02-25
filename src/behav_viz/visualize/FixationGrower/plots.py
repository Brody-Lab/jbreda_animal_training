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
    make_fixation_delta_df,
    compute_failed_fixation_rate_df,
    filter_failed_fix_df,
)
from behav_viz.visualize.df_preperation import compute_days_relative_to_stage

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


def plot_cpoke_dur_distributions(
    trials_df, ax=None, xmin=None, xmax=None, title=None, **kwargs
):
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

    if xmin:
        plot_df = plot_df.query("relative_cpoke_dur > @xmin")
    if xmax:
        plot_df = plot_df.query("relative_cpoke_dur < @xmax")

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

    # to avoid binning issues
    if plot_df.empty:
        return
    elif len(plot_df) < 30:
        binwidth = None
    else:
        binwidth = 0.025

    sns.histplot(
        data=plot_df,
        x="relative_cpoke_dur",
        binwidth=binwidth,
        hue="was_valid",
        palette=pal,
        hue_order=[False, True],
        ax=ax,
        **kwargs,
    )

    if failure_rate != 0:
        avg_failed_dur = plot_df.query("was_valid == False").relative_cpoke_dur.mean()
        ax.axvline(avg_failed_dur, color=pal[0], lw=3, label="Avg Failed Duration")
    if failure_rate <= 0.99:
        avg_valid_dur = plot_df.query("was_valid == True").relative_cpoke_dur.mean()
        ax.axvline(avg_valid_dur, color=pal[1], lw=3, label="Avg Valid Duration")

    avg_cpoke_dur = plot_df.cpoke_dur.mean()

    ax.axvline(0, color="k")

    if title is None:
        title = f"Failure_rate {failure_rate:.2f},  Avg Cpoke Dur: {avg_cpoke_dur:.2f}"

    # aesthetics
    _ = ax.set(
        xlabel="Fixation Dur Relative to Go [s]",
        title=title,
        ylim=(0, None),
        xlim=(xmin, xmax),
    )

    return None


def plot_cpoke_dur_distributions_prettier(
    trials_df, ax=None, xmin=None, xmax=None, title=None, title_color="k", **kwargs
):
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

    if xmin:
        plot_df = plot_df.query("relative_cpoke_dur > @xmin")
    if xmax:
        plot_df = plot_df.query("relative_cpoke_dur < @xmax")

    # Settings
    if settling_in_determines_fix:
        pal = ["#b89f32", "#1762bd"]
        failure_rate = np.sum(trials_df.n_settling_ins > 1) / len(trials_df)
    else:
        pal = ["#bd7217", "#1762bd"]
        failure_rate = trials_df.violations.mean()

    # Plot
    if ax is None:
        _, ax = pu.make_fig()

    # to avoid binning issues
    if plot_df.empty:
        return
    elif len(plot_df) < 30:
        binwidth = None
    else:
        binwidth = 0.025

    sns.histplot(
        data=plot_df,
        x="relative_cpoke_dur",
        binwidth=binwidth,
        hue="was_valid",
        palette=pal,
        hue_order=[False, True],
        ax=ax,
        **kwargs,
    )

    if failure_rate != 0:
        avg_failed_dur = plot_df.query("was_valid == False").relative_cpoke_dur.mean()
        ax.axvline(avg_failed_dur, color=pal[0], lw=3, label="Avg Failed Duration")
    if failure_rate <= 0.99:
        avg_valid_dur = plot_df.query("was_valid == True").relative_cpoke_dur.mean()
        ax.axvline(avg_valid_dur, color=pal[1], lw=3, label="Avg Valid Duration")

    avg_cpoke_dur = plot_df.cpoke_dur.mean()

    ax.axvline(0, color="k")

    if title is None:
        title = f"Failure_rate {failure_rate:.2f},  Avg Cpoke Dur: {avg_cpoke_dur:.2f}"

    ax.legend(title="", loc="upper left")

    # aesthetics
    _ = ax.set(
        xlabel="Fixation Dur Relative to Go [s]",
        ylim=(0, None),
        xlim=(xmin, xmax),
    )
    ax.set_title(title, color=title_color, fontweight="bold")

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

    if settling_in_determines_fix & len(trials_df) > 1:
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

        # sometimes there are no violations, so we can't make this plot
        if data.Viol.isna().all() or data.empty:
            return
        elif len(data) < 30:
            binwidth = None
            avg_viol_cpoke = np.nan
        else:
            # only plot vlines if there is enough data
            binwidth = 0.025
            # Plot avg failed cpoke
            avg_viol_cpoke = trials_df.query("violations == 1").cpoke_dur.mean()
            ax.axvline(avg_viol_cpoke, color=pal[1], lw=3)

            # Plot avg failed settling in
            avg_settling_in = trials_df.avg_settling_in.mean()
            ax.axvline(avg_settling_in, color=pal[0], lw=3)

        sns.histplot(data=data, binwidth=binwidth, palette=pal, ax=ax)

        ax.axvline(trials_df.fixation_dur.mean(), color="k", lw=3)

        ax.set(
            xlabel="Failed Cpoke Dur [s]",
            title=f"Avg dur Viol {avg_viol_cpoke}",
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

    if len(trials_df.query("violations == 0")) == 0:
        # no valid cpokes to plot
        return
    elif len(trials_df.query("violations == 0")) < 30:
        binwidth = None
    else:
        binwidth = 0.025

    sns.histplot(
        trials_df.query("violations == 0").cpoke_dur,
        color="lightgreen",
        binwidth=binwidth,
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
        palette=["lightgreen", "orangered"],
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

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    pu.set_legend(ax, legend)  # ax.legend(["Valid", "Invalid", "Fix", "Init"])
    if legend:
        ax.legend(title="Was Valid")
    _ = ax.set(title=title, ylabel="Duration [s]", xlabel="", ylim=(0, None))
    ax.grid()

    return None


def plot_cpoke_fix_stats_relative(
    trials_df,
    ax=None,
    title="Relative Fixation Stats",
    rotate_x_labels=True,
    legend=True,
    relative_to_stage=None,
    max_days=None,
    **kwargs,
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
    if relative_to_stage:
        x_var = f"days_relative_to_stage_{relative_to_stage}"
        xlabel = f"Days rel to stage {relative_to_stage}"
    else:
        x_var = "date"
        xlabel = ""

    # compute center poking stats
    plot_df = make_long_cpoking_stats_df(
        trials_df, relative=True, relative_to_stage=relative_to_stage
    )

    if max_days and relative_to_stage:
        plot_df = plot_df.query(
            f"days_relative_to_stage_{relative_to_stage} <= @max_days"
        )

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=plot_df,
        x=x_var,
        y="relative_cpoke_dur",
        hue="was_valid",
        hue_order=[True, False],
        palette=["lightgreen", "orangered"],
        marker=".",
        ax=ax,
        **kwargs,
    )
    sns.lineplot(
        data=plot_df,
        x=x_var,
        y="relative_fixation_dur",
        color="gray",
        marker=".",
        ax=ax,
        **kwargs,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    pu.set_legend(ax, legend)  # ax.legend(["Valid", "Invalid", "Fix", "Init"])
    if legend:
        ax.legend(title="Was Valid")

    _ = ax.set(title=title, ylabel="Duration [s]", xlabel=xlabel)
    ax.grid()

    return None


def plot_cpoke_fix_stats_relative_prettier(
    trials_df,
    ax=None,
    title="Relative Fixation Stats",
    rotate_x_labels=True,
    legend=True,
    relative_to_stage=None,
    max_days=None,
    title_color="black",
    **kwargs,
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
    if relative_to_stage:
        x_var = f"days_relative_to_stage_{relative_to_stage}"
        xlabel = f"Days rel to stage {relative_to_stage}"
    else:
        x_var = "date"
        xlabel = ""

    # compute center poking stats
    plot_df = make_long_cpoking_stats_df(
        trials_df, relative=True, relative_to_stage=relative_to_stage
    )

    if max_days and relative_to_stage:
        plot_df = plot_df.query(
            f"days_relative_to_stage_{relative_to_stage} <= @max_days"
        )

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=plot_df,
        x=x_var,
        y="relative_cpoke_dur",
        hue="was_valid",
        hue_order=[True, False],
        palette=["#1762bd", "#bd7217"],
        ax=ax,
        **kwargs,
    )

    ax.axhline(0, color="k")

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    pu.set_legend(ax, legend)  # ax.legend(["Valid", "Invalid", "Fix", "Init"])
    if legend:
        ax.legend(title="Was Valid Cpoke")

    _ = ax.set(ylabel="Time Relative to Go [s]", xlabel=xlabel)
    _ = ax.set_title(title, color=title_color, fontweight="bold")
    ax.grid()

    return None


############################ FIXATION DELTA ######################################


def plot_delta_fixation_dur(trials_df, ax=None, title="", rotate_x_labels=False):
    """

    Plot the delta fixation dur over days. This is computed as the
    difference between the max fixation durs on consecutive days.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date` and `fixation_dur`
        with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    if ax is None:
        fig, ax = pu.make_fig()

    max_fixation_df = make_fixation_delta_df(trials_df)

    if max_fixation_df.empty:
        return

    # check to make sure there has been > 1 day of fixation data
    # to be able to have a diff
    if len(max_fixation_df) == 1:
        return

    sns.lineplot(
        data=max_fixation_df,
        x="date",
        y="fixation_delta",
        ax=ax,
        marker="o",
        color="#B19220",
    )

    ax.axhline(
        max_fixation_df.fixation_delta.mean(),
        color="#B19220",
        linestyle="--",
        alpha=0.5,
    )

    ax.axhline(0, color="k")

    # aesthetics
    min_delta = max_fixation_df.fixation_delta.min() - 0.1
    max_delta = max_fixation_df.fixation_delta.max() + 0.1
    if rotate_x_labels:
        plt.xticks(rotation=45)
    _ = ax.set(
        title=title + f" Avg: {max_fixation_df.fixation_delta.mean():.2f}",
        ylabel="$\Delta$ Fix Dur [s]",
        xlabel="",
        ylim=(min(min_delta, 0), max_delta),
    )
    ax.grid()

    return None


def plot_fixation_dur_box_plot(trials_df, ax=None, title="", rotate_x_labels=False):
    """ """
    if ax is None:
        fig, ax = pu.make_fig()

    flierprops = dict(marker=".", markersize=1.5, linestyle="none")

    sns.boxplot(
        x="date",
        y="fixation_dur",
        data=trials_df,
        ax=ax,
        color="#B19220",
        flierprops=flierprops,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(title=title, ylabel="Fixation Dur [s]", xlabel="", ylim=(0, None))


############################ FAILED FIX (VIOLATIONS) ######################################


def plot_failed_fixation_rate(
    trials_df,
    settling_in_type,
    min_stage=None,
    max_stage=None,
    relative_to_stage=None,
    ax=None,
    title="",
    rotate_x_labels=False,
    **kwargs,
):
    """
    Plot the failed fixation rate over days. This function determines
    how the failed fixation rate is calculated based on the settling in
    determines fixation mode. If on, then failed fixation rates are the
    rate of trials with failed settling ins (by_trial). Or rate of pokes
    with failed cpoke (by_poke). If off, this is simply the violation rate

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date` and `violations`
        with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    # Prepare Data
    failed_fix_df = compute_failed_fixation_rate_df(trials_df)

    failed_fix_df = filter_failed_fix_df(
        failed_fix_df, min_stage, max_stage, settling_in_type
    )

    if failed_fix_df.empty:
        return

    if relative_to_stage:
        failed_fix_df = compute_days_relative_to_stage(
            failed_fix_df, relative_to_stage
        ).reset_index()
        x_var = f"days_relative_to_stage_{relative_to_stage}"
        xlabel = f"Days rel to stage {relative_to_stage}"
    else:
        x_var = "date"
        xlabel = ""

    # Plot
    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        failed_fix_df,
        x=x_var,
        y="failure_rate",
        marker="o",
        ax=ax,
        **kwargs,
    )
    # Handle legend for hue only

    hue = kwargs.get("hue", None)
    style = kwargs.get("style", None)
    if hue and style:
        handles, labels = ax.get_legend_handles_labels()
        cutoff_index = labels.index(style)
        hue_handles = handles[:cutoff_index]
        hue_labels = labels[:cutoff_index]
        ax.legend(hue_handles, hue_labels)

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(
        ylabel="Failed Fixation Rate", xlabel=xlabel, title=title, ylim=(-0.1, 1.1)
    )
    ax.grid()

    return None


def plot_failed_fixation_rate_ods(
    trials_df,
    ax,
    title,
):
    """
    Quick wrapper function specific to the over days summary plot that
    passes in the desired formatting for the plot.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date` and `violations`
        `settling_in_determines_fixation` with trials as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str
        title for the plot
    """

    plot_failed_fixation_rate(
        trials_df,
        settling_in_type="all",
        ax=ax,
        title=title,
        rotate_x_labels=False,
        hue="type",
        hue_order=["by_trial", "by_poke", "violation"],
        palette=["plum", "purple", "orangered"],
    )

    return None


############################ TRIAL STRUCTURE ######################################


def plot_trial_structure(
    trials_df, ax=None, kind="bar", title="", legend=True, rotate_x_labels=False
):
    """
    Plot the trial structure over date range in trials_df. Note that this
    assumes that the stimuli are on and playing.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date`, `settling_in_dur`, `adj_pre_dur`,
        `stimulus_dur`, `delay_dur`, `post_dur` with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    kind : str (optional, default = "bar")
        kind of plot to make
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    stim_on = trials_df.stimuli_on.iloc[-1] > 0

    if stim_on:
        columns_to_plot = [
            "date",
            "settling_in_dur",
            "adj_pre_dur",
            "stimulus_dur",
            "delay_dur",
            "post_dur",
        ]
    else:
        columns_to_plot = ["date", "settling_in_dur", "pre_go_dur"]

    day_avgs = trials_df[columns_to_plot].groupby("date").mean().reset_index()
    day_avgs.columns = day_avgs.columns.str.replace("_dur", "")
    if stim_on:
        day_avgs.insert(5, "s_b", day_avgs["stimulus"])
        day_avgs.rename(columns={"stimulus": "s_a"}, inplace=True)
    else:
        # rename pre go to delay as this is the delay duration adjusted
        # for settling in dur
        day_avgs.rename(columns={"pre_go": "delay"}, inplace=True)

    if ax is None:
        fig, ax = pu.make_fig()

    day_avgs.plot(
        x="date",
        kind=kind,
        stacked=True,
        ax=ax,
        legend=legend,
        color=pu.trial_period_palette,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(title=title, xlabel="", ylabel="Trial Timing [s]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5, axis="y")
    ax.legend(loc="upper left")

    return None


############################ FIXATION DUR ######################################


def plot_fixation_dur_over_days(
    df,
    min_stage=None,
    max_stage=None,
    relative_to_stage=5,
    ax=None,
    title="",
    rotate_x_labels=False,
    **kwargs,
):

    df = compute_days_relative_to_stage(df, relative_to_stage)
    x_var = f"days_relative_to_stage_{relative_to_stage}"
    xlabel = f"Days rel to stage {relative_to_stage}"

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=df.query("stage >= @min_stage and stage <= @max_stage"),
        x=x_var,
        y="fixation_dur",
        estimator="max",
        ax=ax,
        **kwargs,
    )

    ax.grid()
    ax.axhline(2, color="k", lw=2)

    _ = ax.set(ylabel="Fix Dur [s]", xlabel=xlabel, title=title)

    if rotate_x_labels:
        ax.tick_params(rotation=0)

    return None
