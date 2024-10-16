"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting functions for looking at training 
performance, water and mass metrics across days
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu

# TODO change to import plot_utils as pu and add into below
#  TODO move plots from `DMS_multiday_plots` to here and update df to be trials_df
# july 2023 note, looks like this might be already done?
#######################
###  SUMMARY PLOTS  ###
#######################


def plot_multiday_summary(animal_id, days_df):
    """
    Plot the summary of the animal's performance over the
    date range in days_df

    params
    ------
    animal_id : str
        animal id to plot, e.g. "R610"
    days_df : pd.DataFrame
        days dataframe created by create_days_df_from_dj()
    """

    layout = """
        AAABBB
        CCCDDD
        EEEFFF
    """
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
    plt.suptitle(f"{animal_id} Daily Summary Plot", fontweight="semibold")

    animal_df = days_df.query("animal_id == @animal_id")

    ## Plot
    # left column
    plot_trials(animal_df, ax_dict["A"], title="Trials", legend=True)
    plot_performance(animal_df, ax_dict["C"], title="Performance")
    plot_side_bias(animal_df, ax_dict["E"], title="Side Bias")

    # right column
    plot_mass(animal_df, ax_dict["B"], title="Mass")
    plot_water_restriction(animal_df, ax_dict["D"], title="Water", legend=False)
    plot_rig_tech(animal_df, ax_dict["F"], title="Rig/Tech")

    return None


######################
###  SINGLE PLOTS  ###
######################


### TRIALS ###
def plot_trials(days_df, ax=None, title="", legend=False, rotate_x_labels=False):
    """
    Plot the number of trials completed and trial rate over
    date range in d_days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `n_done_trials`, `trial_rate` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on (optional, default = None)
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        fig, ax = pu.make_fig()

    trial_melt = days_df.melt(
        id_vars=["date"],
        value_name="trial_var",
        value_vars=["n_done_trials", "trial_rate"],
    )
    sns.lineplot(
        data=trial_melt,
        x="date",
        y="trial_var",
        hue="variable",
        marker="o",
        ax=ax,
    )

    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    pu.set_legend(ax, legend)
    _ = ax.set(ylabel="Count || Per Hr", xlabel="", title=title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)

    return None


### STAGE ###
def plot_stage(
    trials_df,
    ax=None,
    title="",
    group="date",
    ylim=None,
    rotate_x_labels=False,
    **kwargs,
):
    """
    Plot stage over group variable.

    Having the group variable allows you to group by other
    things like "start_date" if you want to try and make a plot
    with multiple animals that started at different times.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date`, `stage` with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    group : str (optional, default = "date")
        column to group by for x axis (e.g. "date", "start_date")
    ylim : tuple (optional, default = None)
        y-axis limits
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=trials_df.groupby(group).stage.mean(),
        drawstyle="steps-post",
        ax=ax,
        marker="o",
        **kwargs,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    if ylim:
        ylim = ylim
        yticks = range(ylim[0], ylim[1] + 1)
    else:
        max_stage = int(trials_df.stage.max())
        ylim = (0, max_stage + 1)
        yticks = range(max_stage + 1)

    ax.set(
        ylabel="Stage #",
        title=title,
        ylim=ylim,
        yticks=yticks,
    )
    ax.grid()


### MASS ###
def plot_mass(days_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the mass of the animal over date range in days_df. If it's a mouse,
    will also plot mass relative to baseline as a percent.

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `mass` with dates as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    baseline_mass = get_baseline_mass(days_df)

    if ax is None:
        fig, ax = pu.make_fig()

    if baseline_mass is np.nan:
        plot_raw_mass(days_df, ax, title=title, rotate_x_labels=rotate_x_labels)
    else:
        plot_raw_and_relative_mass(
            days_df, baseline_mass, ax, title=title, rotate_x_labels=rotate_x_labels
        )

    return None


def get_baseline_mass(days_df):
    """
    Determine if animal is a mouse or rat, so baseline mass plot
    can be determined. This assumes you have an "ANIMALS_TABLE",
    which is located in the data directory. I manually enter high-level
    animal info here that's not easy to access/store on datajoint.

    params
    ------
    days_df : pd.DataFrame
        days dataframe with `animal_id` column
    """

    from behav_viz.utils.dir_utils import ANIMAL_TABLE_PATH

    animal_id = days_df.animal_id.unique()[0]
    animal_table = pd.read_excel(ANIMAL_TABLE_PATH)
    # species = animal_table.query("animal_id == @animal_id").species.iloc[0]

    # if species == "mouse":
    #     baseline_mass = animal_table.query(
    #         "animal_id == @animal_id"
    #     ).baseline_mass.iloc[0]
    # else:
    baseline_mass = np.nan

    return baseline_mass


def plot_raw_mass(days_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the raw mass in grams over date range in days_df.

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `mass` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(data=days_df, x="date", y="mass", marker="o", color="k", ax=ax)

    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.5)
    ax.set(ylabel="Mass [g]", xlabel="", title=title)

    return None


def plot_raw_and_relative_mass(
    days_df, baseline_mass, ax=None, title="", rotate_x_labels=False
):
    """
    Plot the raw & relative mass of an animal over the
    date range in days_df. The axes are locked to each other
    so that you can easily convert between the two.

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `mass` with dates as row index
    baseline_mass : float
        baseline mass of the mouse in grams, calculated however
        you like bust most easily by get_baseline_mass()
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    def _mass_to_relative(mass, baseline_mass):
        return mass / baseline_mass * 100

    def _convert_ax_r_to_relative(ax_m):
        """
        update second axis according to first axis, such that
        mass -> relative mass
        """

        y1, y2 = ax.get_ylim()
        ax_r.set_ylim(
            _mass_to_relative(y1, baseline_mass), _mass_to_relative(y2, baseline_mass)
        )
        ax_r.figure.canvas.draw()

    if ax is None:
        fig, ax = pu.make_fig()

    # make the baseline mass on the right y-axis always match
    # the raw mass on the left y-axis
    ax_r = ax.twinx()
    ax.callbacks.connect("ylim_changed", _convert_ax_r_to_relative)
    ax.plot(days_df["date"], days_df["mass"], color="k", marker="o")
    ax.axhline(baseline_mass * 0.8, color="red", linestyle="--", alpha=0.5)

    # aesthetics
    ax.set(
        ylim=(baseline_mass * 0.75, baseline_mass),
        ylabel="Mass [g]",
    )
    ax_r.set(ylabel="Relative Mass [%]", title=title)

    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.grid()

    return None


### WATER ###
def plot_water_restriction(
    days_df,
    ax=None,
    title="",
    legend=True,
    rotate_x_labels=False,
):
    """
    Plot the rig, pub and restriction target volume over date
    range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `rig_volume`, `pub_volume`
        and `volume_target` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on (optional, default = None)
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    # stacked bar chart only works with df.plot (not seaborn)
    # if date is datetime convert to date only
    if days_df.date.dtype == "datetime64[ns]":
        days_df["date_only"] = days_df["date"].dt.date
        date_col = "date_only"
    else:
        date_col = "date"
    columns_to_plot = [date_col, "rig_volume", "pub_volume"]

    days_df[columns_to_plot].plot(
        x=date_col,
        kind="bar",
        stacked=True,
        color=["blue", "cyan"],
        ax=ax,
    )

    if ax is None:
        fig, ax = pu.make_fig()
    # iterate over dates to plot volume target black line
    for i, row in days_df.reset_index().iterrows():
        # if percent target is 20, then pub is effectively off
        if row["percent_target"] == 20:
            continue
        else:
            ax.hlines(
                y=row["volume_target"], xmin=i - 0.35, xmax=i + 0.35, color="black"
            )
    pu.set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel="Volume [mL]")

    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    return None


### RIG/TECH ###
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


### PERFORMANCE ###
def plot_performance(days_df, ax=None, title="", legend=True, rotate_x_labels=False):
    """
    Plot the hit and violation rate over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `hit_rate` and `viol_rate` with
        dates as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=days_df,
        x="date",
        y="hit_rate",
        marker="o",
        color="darkgreen",
        label="hit",
        ax=ax,
    )
    sns.lineplot(
        data=days_df,
        x="date",
        y="viol_rate",
        marker="o",
        color="orangered",
        label="viol",
        ax=ax,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    pu.set_legend(ax, legend)
    ax.grid(alpha=0.5)
    ax.set(ylim=(0, 1), ylabel="Perf Rate", xlabel="", title=title)

    return None


def plot_performance_bars(
    trials_df,
    ax=None,
    plot_type="counts",
    title="",
    legend=False,
    rotate_x_labels=False,
):
    """
    Plot the count or rate of results in a stacked bar over date
    range in df

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and `result`
        with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    plot_type : str, (default = "counts")
        whether to plot counts or rates
    title : str, (default = "")
        title of plot
    legend : bool, (default = True)
        whether to include legend or not
    rotate_x_labels : bool, (default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        fig, ax = pu.make_fig()

    if plot_type == "counts":
        perf_df = trials_df.groupby(["date"]).result.value_counts().unstack()
    elif plot_type == "rates":
        perf_df = (
            trials_df.groupby(["date"]).result.value_counts(normalize=True).unstack()
        )
    else:
        raise ValueError("plot type can only be 'counts' or 'rates'")
    perf_df.plot(
        kind="bar", stacked=True, ax=ax, color=pu.get_result_colors(perf_df.columns)
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    pu.set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel=f"Perf {plot_type}")

    return None


def plot_performance_w_error(
    trials_df, ax=None, title="", legend=False, rotate_x_labels=False
):
    """
    Plot the hit violation and error rate over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `hit_rate` `error_rate` and
        `viol_rate` with dates as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    perf_rates_df = pd.melt(
        trials_df,
        id_vars=["date"],
        value_vars=["violation_rate", "error_rate", "hit_rate"],
    )

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=perf_rates_df,
        x="date",
        y="value",
        hue="variable",
        palette=["orangered", "maroon", "darkgreen"],
        errorbar=None,
        marker="o",
        ax=ax,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    pu.set_legend(ax, legend)
    ax.grid(alpha=0.5)
    _ = ax.set(ylabel="Perf Rate", xlabel="", title=title, ylim=(0, 1))

    return None


### SIDE BIAS ###
def plot_side_bias(days_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the side bias over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date` and `side_bias` with
        dates as row index, positive values = right bias
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
        data=days_df,
        x="date",
        y="side_bias",
        color="lightseagreen",
        marker="o",
        ax=ax,
    )
    ax.axhline(0, color="k", linestyle="--", zorder=1)

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.5)
    ax.set(ylim=(-1, 1), ylabel="< - Left | Right ->", xlabel="", title=title)

    return None


def plot_antibias_probs(
    trials_df, ax=None, title="", legend=True, rotate_x_labels=False
):
    """
    Plot the left and right antibias probabilities over date range in trials_df

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date`, `ab_l_prob` and `ab_r_prob`
        with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not

    """
    ab_melt = trials_df.melt(
        id_vars=["date"], value_vars=["ab_l_prob", "ab_r_prob"], value_name="antibias"
    )

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=ab_melt,
        x="date",
        y="antibias",
        hue="variable",
        marker="o",
        palette=["darkseagreen", "indianred"],
        ax=ax,
    )
    ax.axhline(0.5, color="k", linestyle="--", zorder=1)

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.set(title=title, xlabel="", ylabel="Prob", ylim=(0, 1))
    ax.legend(frameon=False, borderaxespad=0)

    return None


def plot_sidebias_params(
    trials_df, ax=None, title="", legend=False, rotate_x_labels=False
):
    """
    Plot the side bias parameters over date range in trials_df. The parameters
    are the left and right water volumes and the antibias beta. The antibias beta
    will have std error if beta warm up is on which means over tau trials (usually
    defaults to 30) the beta goes from 0 to the set value.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date`, `l_water_vol`, `r_water_vol` and
        `ab_beta` with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    sidebias_melt = trials_df.melt(
        id_vars=["date"],
        value_name="sb_vars",
        value_vars=["l_water_vol", "r_water_vol", "ab_beta"],
    )

    if ax is None:
        fig, ax = pu.make_fig()

    sns.barplot(
        data=sidebias_melt,
        x="date",
        y="sb_vars",
        hue="variable",
        palette=["teal", "purple", "blue"],
        alpha=0.5,
        ax=ax,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    pu.set_legend(ax, legend)
    ax.grid(axis="y")
    _ = ax.set(title=title, xlabel="", ylabel="Value")

    return None


### SIDE POKE ###
def plot_time_to_spoke(
    trials_df, ax=None, title="", legend=True, rotate_x_labels=False
):
    """
    Plot the time to the first L and R poke for each trial over
    date range in trials_df

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date`, `min_time_to_spoke` and `first_spoke`
        with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=trials_df,
        x="date",
        y="min_time_to_spoke",
        hue="first_spoke",
        hue_order=pu.get_side_order(trials_df.first_spoke),
        marker="o",
        palette=pu.get_side_colors(trials_df.first_spoke),
        ax=ax,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    pu.set_legend(ax, legend)
    ax.set(ylabel="time to spoke [s]", xlabel="", title=title, ylim=(0))
    ax.grid(alpha=0.5)

    return None


### CPOKE ###


def plot_n_cpokes_and_multirate(trials_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the number of center pokes and the multi cpoke rate over
    date range in trials_df. Not this is specific to DMS2 where
    violation penalties aren't on.

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date`, `n_settling_ins` with trials as row index
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
        y="n_settling_ins",
        ax=ax,
        marker="o",
        label="N Cpokes",
    )

    # aesthetics for left axis
    ax.set_ylim(bottom=0)
    ax.set(ylim=(0), ylabel=("Avg N Cpokes / Trial"), xlabel="", title=title)

    ax2 = ax.twinx()

    trials_df["multi_cpoke"] = trials_df["n_settling_ins"] > 1
    trials_df.groupby("date").multi_cpoke.mean().plot(
        kind="line", ax=ax2, marker="o", label="mutli cpoke rate", color="orange"
    )
    # aesthetics for right axis
    ax2.set(ylim=(-0.1, 1), ylabel="Multi Cpoke Rate", xlabel="")
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.grid()

    return None


def plot_cpoke_dur_timings_pregnp(trials_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the duration of the center poke and the timings of the
    the valid and non valid center pokes over date range in trials_df
    This is specific to the DMS2 task. For each day this will plot, the
    average settling in duration, the average valid center poke duration
    and the average non valid center poke duration (violation).

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `date`, `avg_settling_in`, `cpoke_dur`
        and `violations` with trials as row index
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
        y="avg_settling_in",
        marker="o",
        ax=ax,
        color="blue",
        label="Settling",
    )

    sns.lineplot(
        data=trials_df.query("violations == 0"),
        x="date",
        y="cpoke_dur",
        marker="o",
        ax=ax,
        color="lightgreen",
        label="Valid",
    )
    try:
        sns.lineplot(
            data=trials_df.query("violations == 1"),
            x="date",
            y="cpoke_dur",
            marker="o",
            ax=ax,
            color="orangered",
            label="Viol",
        )
    except:  # not enough data to plot
        pass

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="Duration [s]", xlabel="", title=title)
    ax.grid()

    return None


### DELAY ###


def plot_exp_delay_params(trials_df, ax, title="", legend=False, xaxis_label=True):
    """
    TODO
    """
    sns.lineplot(data=trials_df, x="date", y="delay_dur", legend=legend, ax=ax)

    delay_melt = trials_df.melt(
        id_vars=["date"],
        value_name="delay_params",
        value_vars=["exp_del_max", "exp_del_tau", "exp_del_min"],
    )

    sns.lineplot(
        data=delay_melt,
        x="date",
        y="delay_params",
        hue="variable",
        palette="gray",
        errorbar=None,
        ax=ax,
    )

    # aesthetics
    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
    _ = ax.set(title=title, xlabel="", ylabel="Delay Dur [s]")

    return None


def plot_avg_delay(trials_df, ax, title="", xaxis_label=True):
    """
    TODO
    """
    sns.lineplot(data=trials_df, x="date", y="delay_dur", marker="o", ax=ax)

    # aesthetics
    pu.set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(title=title, xlabel="", ylabel="Avg Delay [s]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)

    return None


### TIMING ###


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
    columns_to_plot = [
        "date",
        "settling_in_dur",
        "adj_pre_dur",
        "stimulus_dur",
        "delay_dur",
        "post_dur",
    ]

    day_avgs = trials_df[columns_to_plot].groupby("date").mean().reset_index()
    day_avgs.insert(5, "s_b", day_avgs["stimulus_dur"])
    day_avgs.rename(columns={"stimulus_dur": "s_a"}, inplace=True)
    day_avgs.columns = day_avgs.columns.str.replace("_dur", "")

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


def plot_trial_end_timing(
    trials_df, ax, kind="bar", title="", legend=True, xaxis_label=True
):
    """
    TODO
    """
    columns_to_plot = ["date", "viol_off_dur", "pre_go_dur"]
    day_avgs = trials_df[columns_to_plot].groupby("date").mean().reset_index()
    day_avgs.plot(x="date", kind="bar", stacked=False, ax=ax, legend=legend)

    # aesthetics
    pu.set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(title=title, xlabel="", ylabel="Trial Timing [s]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5, axis="y")

    return None


### GIVE ###


def plot_performance_by_give(
    trials_df,
    ax=None,
    title="",
    group="date",
    legend=True,
    rotate_x_labels=False,
):
    """
    generate a plot of hit rate for non-give trials

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `hits` and
        `give_type_imp` with trials as row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str, (default = "")
        title of plot
    group : str, (default = "date")
        column to group by for x axis
    legend : bool (optional, default = True)
        whether to include the legend or not


    """

    sns.lineplot(
        data=trials_df,
        x=group,
        y="hits",
        marker="o",
        hue="give_type_imp",
        palette=pu.get_give_colors(trials_df["give_type_imp"]),
        hue_order=pu.get_give_order(trials_df["give_type_imp"]),
        ax=ax,
    )

    # mark number of trials with give
    give_proportions = (
        trials_df.groupby(group)
        .give_type_imp.value_counts(normalize=True)
        .reset_index()
    )

    if ax is None:
        fig, ax = pu.make_fig()

    sns.lineplot(
        data=give_proportions.query("give_type_imp != 'none'"),
        x=group,
        y="proportion",
        marker="x",
        ax=ax,
        color="black",
    )
    ax.axhline(0.6, color="gray", linestyle="--")

    # aethetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(ylabel="Proportion", xlabel="", title=title, ylim=(0, 1))
    pu.set_legend(ax, legend)
    ax.grid(alpha=0.5)

    return None


def plot_give_info_days(
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
    mapping = {"water_and_light": "w + l", "water": "w", "light": "l", "none": "n"}
    data.give_type_imp = data.give_type_imp.replace(mapping)

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

    return None


### SOUNDS ###


def plot_sounds_info(trials_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the sound volume and duration info across days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `volume_multiplier` `stimulus_dur` with trials as
        row index
    ax : matplotlib.axes.Axes (optional, default = None)
        axes to plot on
    title : str, (default = "")
        title of plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    # find minimum volume and stimulus dur for each day
    plot_df = (
        trials_df.melt(
            id_vars=["date", "animal_id"],
            value_vars=["volume_multiplier", "stimulus_dur"],
            var_name="sound_var",
            value_name="value",
        )
        .groupby(["date", "sound_var"])
        .value.min()
        .reset_index()
    )

    if ax is None:
        fig, ax = pu.make_fig()
    # plot
    sns.lineplot(
        data=plot_df,
        x="date",
        y="value",
        hue="sound_var",
        marker="o",
        palette=["purple", "cornflowerblue"],
        ax=ax,
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(title=title, ylabel="Sound Variable Value", xlabel="")
    ax.legend(loc="center left")

    return None


def plot_performance_by_stim_over_days(
    trials_df,
    without_give,
    ax=None,
    confidence_intervals=True,
    x_var="date",
    title="",
    rotate_x_labels=False,
):
    """
    Plot performance by sa, sb pair over days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date`, `sound_pair`,
        `give_use` and `hits` with trials as row index
    without_give : bool
        whether to plot the performance of trials without give
        or not
    ax : matplotlib.axes.Axes
        axes to plot on
    confidence_intervals : bool (optional, default = True)
        whether to plot the 95% confidence intervals or not
    x_var : str (default='date')
        variable to plot on x axis
    title : str, (default = "")
        title of plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    if ax is None:
        fig, ax = pu.make_fig()

    if without_give:
        df = trials_df.query("give_use == 0").copy()
    else:
        df = trials_df.copy()
    df["hits"] = df["hits"].astype("float64")

    # Calculate mean and confidence interval (95% CI as example)
    perf_mean = df.pivot_table(
        index=x_var, columns="sound_pair", values="hits", aggfunc="mean"
    )
    if confidence_intervals:
        perf_sem = df.pivot_table(
            index=x_var, columns="sound_pair", values="hits", aggfunc="sem"
        )  # Standard Error of the Mean
        confidence = 1.96 * perf_sem  # Approx. for 95% CI, assuming normal distribution

    colors = {
        "3.0, 3.0": "skyblue",
        "12.0, 12.0": "steelblue",
        "3.0, 12.0": "thistle",
        "12.0, 3.0": "mediumorchid",
    }

    # Plot lines and fill confidence intervals
    for sound_pair in perf_mean.columns:
        ax.plot(
            perf_mean.index,
            perf_mean[sound_pair],
            label=sound_pair,
            color=colors[sound_pair],
            linestyle="-",
            marker=".",
        )
        if confidence_intervals:
            ax.fill_between(
                perf_mean.index,
                perf_mean[sound_pair] - confidence[sound_pair],
                perf_mean[sound_pair] + confidence[sound_pair],
                color=colors[sound_pair],
                alpha=0.2,
            )

    ax.axhline(0.6, color="k", linestyle="--")
    ax.legend(loc="lower left")
    ax.set(title=title, xlabel="", ylabel="Hit Rate", ylim=(0, 1))
    ax.grid()
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    return None


def plot_non_give_stim_performance(
    trials_df,
    ax,
    group="date",
    title="",
    xaxis_label=True,
    aesthetics=True,
    variance=False,
):
    """
    !NOT CURRENTLY IN USE see plot_performance_by_stim_over_days()
    Plot performance by sa, sb pair on non-give
    trials across days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date`, `sound_pair`,
        `give_type_imp` and `hits` with trials as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str, (default = "")
        title of plot
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    aesthetics : bool (optional, default = True)
        used to toggle xaxis label when subplotting
    variance : bool (optional, default = False)
        whether or not to plot the variance of the data (95% ci) or not
    """

    # when hits remains in pyarrow format, the groupby
    # doesn't work properly for some edge cases
    sub_df = trials_df.query("give_use == 0").copy()
    sub_df["hits"] = sub_df["hits"].astype("float64")

    if variance:
        plot_df = sub_df.copy()
    else:
        plot_df = sub_df.groupby([group, "sound_pair"]).hits.mean().reset_index()

    sns.lineplot(
        data=plot_df,
        x=group,
        y="hits",
        hue="sound_pair",
        palette=pu.create_palette_given_sounds(plot_df),
        marker="o",
        ax=ax,
    )

    ax.grid()
    ax.axhline(0.6, color="k", linestyle="--")
    ax.set(title=title, xlabel="", ylabel="Hit Rate", ylim=(0, 1))
    if aesthetics:
        pu.set_date_x_ticks(ax, xaxis_label)
    ax.legend(loc="lower left")

    return None


def plot_stim_performance(
    trials_df,
    ax,
    x_var="date",
    title="",
    errorbar=None,
    xaxis_label=True,
    aesthetics=True,
    **kwargs,
):
    """
    !NOT CURRENTLY IN USE see plot_performance_by_stim_over_days()
    Plot performance by sa, sb pair on all trials
    trials across days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date`, `sound_pair`,
        `give_type_imp` and `hits` with trials as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    x_var : str (default='date')
        variable to plot on x axis
    title : str, (default = "")
        title of plot
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """

    sns.lineplot(
        data=trials_df,
        x=x_var,
        y="hits",
        hue="sound_pair",
        palette=pu.create_palette_given_sounds(trials_df),
        marker="o",
        ax=ax,
        errorbar=errorbar,
        **kwargs,
    )

    ax.grid()
    ax.axhline(0.6, color="k", linestyle="--")
    ax.set(title=title, xlabel="", ylabel="Hit Rate", ylim=(0, 1))
    ax.legend(loc="lower left")

    if aesthetics:
        pu.set_date_x_ticks(ax, xaxis_label)

    return None


## PRO ANTI ##
def plot_performance_by_pro_anti_over_days(
    trials_df,
    without_give,
    ax=None,
    confidence_intervals=True,
    x_var="date",
    title="",
    rotate_x_labels=True,
):
    """
    Plot performance by pro-anti over days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date`, `sound_pair`,
        `give_use`, `pro_anti_block_type` and `hits` with trials
        as row index
    without_give : bool
        whether to plot the performance of trials without give
        or not
    ax : matplotlib.axes.Axes
        axes to plot on
    confidence_intervals : bool (optional, default = True)
        whether to plot the 95% confidence intervals or not
    x_var : str (default='date')
        variable to plot on x axis
    title : str, (default = "")
        title of plot
    rotate_x_labels : bool (optional, default = True)
        whether to rotate the x-axis labels or not
    """

    if ax is None:
        fig, ax = pu.make_fig()

    # Filter and prepare data
    if without_give:
        df = trials_df.query("give_use == 0").copy()
    else:
        df = trials_df.copy()
    df["hits"] = df["hits"].astype("float64")

    plot_data = (
        df.query("pro_anti_block_type != 'NA'")
        .dropna(subset=["pro_anti_block_type"])
        .copy()
    )

    # Calculate mean and standard error for the confidence interval
    perf_mean = plot_data.pivot_table(
        index=x_var, columns="pro_anti_block_type", values="hits", aggfunc="mean"
    )
    confidence_intervals = plot_data.pivot_table(
        index=x_var, columns="pro_anti_block_type", values="hits", aggfunc="sem"
    )
    confidence = 1.96 * confidence_intervals  # Assuming 95% CI

    # Define colors
    colors = sns.color_palette("husl", 2)
    color_map = {"pro": colors[0], "anti": colors[1]}

    # Plot lines with pandas plot, iterating to add 95% CIs
    for column in perf_mean.columns:
        ax.plot(
            perf_mean.index,
            perf_mean[column],
            label=column,
            color=color_map[column],
            marker=".",
        )
        ax.fill_between(
            perf_mean.index,
            perf_mean[column] - confidence[column],
            perf_mean[column] + confidence[column],
            color=color_map[column],
            alpha=0.2,
        )

    # Aesthetics
    ax.grid()
    ax.axhline(0.6, color="k", linestyle="--")
    ax.legend(loc="lower left", title="Block Type")
    ax.set(
        title=title,
        xlabel="" if not rotate_x_labels else x_var,
        ylabel="Hit Rate",
        ylim=(0, 1),
    )
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    return None


def plot_stim_performance_by_pro_anti(
    trials_df, ax, x_var="date", title="", xaxis_label=True, aesthetics=True
):
    """
    !NOT IN USE see plot_performance_by_pro_anti_over_days()
    Plot performance by pro or anti trials across days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date`, `sound_pair`,
        `give_type_imp` and `hits` with trials as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    x_var : str (default='date')
        variable to plot on x axis
    title : str, (default = "")
        title of plot
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """

    # TODO once 20 days have passed since imp, add this to create_trials_df.py
    plot_data = (
        trials_df.query("pro_anti_block_type != 'NA'")
        .dropna(subset=["pro_anti_block_type"])
        .copy()
    )

    sns.lineplot(
        data=plot_data,
        x=x_var,
        y="hits",
        hue="pro_anti_block_type",
        hue_order=["pro", "anti"],
        palette="husl",
        marker="o",
        ax=ax,
    )

    ax.grid()
    ax.axhline(0.6, color="k", linestyle="--")
    ax.legend(loc="lower left")
    ax.set(title=title, xlabel="", ylabel="Hit Rate", ylim=(0, 1))
    if aesthetics:
        pu.set_date_x_ticks(ax, xaxis_label)

    return None


def plot_n_pro_anti_blocks_days(
    trials_df, ax=None, x_var="date", title="", rotate_x_labels=False
):
    """
    Plot the number of pro-anti blocks per day

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and `n_blocks`
        with trials as row index
    ax : matplotlib.axes.Axes (default=None)
        axes to plot to
    x_var : str (default='date')
        variable to plot on x axis
    title : str (default='')
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        _, ax = pu.make_fig()

    sns.lineplot(
        data=trials_df,
        x=x_var,
        y="n_blocks",
        estimator="max",
        ax=ax,
        color="k",
        marker="o",
        label="Block Switches",
    )

    if trials_df.stage.max() == 15:
        sns.lineplot(
            data=trials_df,
            x=x_var,
            y="max_blocks",
            color="purple",
            marker="o",
            label="Max Blocks",
            ax=ax,
        )

    # aethetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    _ = ax.set(
        ylabel="N Blocks",
        xlabel="",
        title=title,
        ylim=(0, trials_df.n_blocks.max() + 1),
    )

    return None


def plot_block_switch_thresholds(trials_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot threshold used for switching blocks
    across days (typically for pro-anti stages)

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `pro_anti_hit_thresh`, `pro_anti_viol_thresh`
        with trials as row index
    ax : matplotlib.axes.Axes, de
        axes to plot on
    title : str, (default=None)
        title of plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not


    """
    thresh_df = pd.melt(
        trials_df,
        id_vars=["date"],
        value_vars=["pro_anti_hit_thresh", "pro_anti_viol_thresh"],
        var_name="type",
        value_name="switch_thresh",
    )

    if ax is None:
        _, ax = pu.make_fig()

    sns.lineplot(
        data=thresh_df,
        x="date",
        y="switch_thresh",
        hue="type",
        hue_order=["pro_anti_hit_thresh", "pro_anti_viol_thresh"],
        palette=["green", "orangered"],
        marker="o",
        ax=ax,
    )

    ax.grid()
    ax.legend(loc="lower left")
    _ = ax.set(title=title, xlabel="", ylabel="Switch Threshold", ylim=(0, 1))
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    return None


def plot_block_switch_days(
    trials_df, ax=None, title="", xaxis_label=True, legend=False
):
    """
    Plot the type of block switch being used
    across days (typically for pro-anti stages)

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `block_switch_type` with trials as row index
    ax : matplotlib.axes.Axes, de
        axes to plot on
    title : str, (default=None)
        title of plot
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    legend : bool (optional, default = True)
        whether to include the legend or not
    """
    # make the names shorter for plotting
    data = trials_df[["date", "block_switch_type"]].copy()

    if ax is None:
        _, ax = pu.make_fig()

    sns.scatterplot(
        data=data,
        x="date",
        y="block_switch_type",
        hue="block_switch_type",
        hue_order=["sampled", "static", "none"],
        ax=ax,
    )

    # aesthetics
    _ = ax.set(title=title, ylabel="", xlabel="")
    pu.set_legend(ax, legend)
    pu.set_date_x_ticks(ax, xaxis_label)

    return None


def plot_min_block_size(
    trials_df, ax=None, x_var="date", title="", rotate_x_labels=False
):
    """
    Plot the block_size parameter that is used to
    determine the minimum number of trials for a switch
    before performance is evaluated over days.

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and `block_size`
        with trials as row index
    ax : matplotlib.axes.Axes (default=None)
        axes to plot to
    x_var : str (default='date')
        variable to plot on x axis
    title : str (default='')
        title for the plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if ax is None:
        _, ax = pu.make_fig()

    sns.lineplot(
        data=trials_df,
        x=x_var,
        y="block_size",
        ax=ax,
        marker="o",
        label="Min Block Size",
        color="gray",
    )

    # aethetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    _ = ax.set(
        ylabel="Min. Block Size",
        xlabel="",
        title=title,
        ylim=(0, trials_df.block_size.max() + 1),
    )
    ax.grid(alpha=0.5)

    return None


def plot_give_type_and_block_switch_days(
    trials_df, ax=None, title="", legend=False, rotate_x_labels=False
):
    """
    Plot the type of block switch and give type being used
    across days (typically for pro-anti stages)

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date`
        `block_switch_type` and `give_type_imp`
        with trials as row index
    ax : matplotlib.axes.Axes, (default=None)
        axes to plot on
    title : str, (default=None)
        title of plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    # make the names shorter for plotting
    data = trials_df[["date", "block_switch_type", "give_type_imp"]].copy()

    data = data.melt(
        id_vars="date",
        value_vars=["block_switch_type", "give_type_imp"],
        var_name="metric",
        value_name="type",
    )

    if ax is None:
        _, ax = pu.make_fig()
    ax.grid()
    sns.scatterplot(
        data=data,
        x="date",
        y="type",
        style="metric",
        hue_order=pu.get_block_switch_and_give_order(),
        palette=pu.get_block_switch_and_give_colors(),
        hue="type",
        s=300,
        ax=ax,
    )

    # aesthetics
    _ = ax.set(title=title, ylabel="Block Switch or Give Type", xlabel="")
    pu.set_legend(ax, legend)
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)

    return None


def plot_block_switch_params(trials_df, ax=None, title="", rotate_x_labels=False):
    """
    Plot the block switch parameters across days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` `block_size`
        and `pro_anti_hit_thresh`, `pro_anti_viol_thresh`
        with trials as row index
    ax : matplotlib.axes.Axes, (default=None)
        axes to plot on
    title : str, (default=None)
        title of plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not

    """
    if ax is None:
        _, ax = pu.make_fig()

    plot_min_block_size(trials_df, ax=ax, rotate_x_labels=rotate_x_labels)
    ax.set_label("Min Block Size")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    plot_block_switch_thresholds(trials_df, ax=ax2, rotate_x_labels=rotate_x_labels)

    ax2.set(
        ylabel="Perf Threshold",
        xlabel="",
        title=title,
        ylim=(0, 1),
        yticks=[0, 0.5, 1],
    )

    return None


## GIVE DELAY PLOTS


def plot_give_delay_dur_days(
    trials_df,
    ax=None,
    trial_subset="anti",
    title="",
    rotate_x_labels=False,
):
    """
    Plot the distribution of pre-give delay durations
    across days.

    params
    ------

    trials_df: pd.DataFrame
        trials dataframe with columns `pro_anti_block_type`,
        `date`, and `give_delay_dur` with trials as row index
    ax : matplotlib.axes.Axes, de
        axes to plot on
    title : str, (default=None)
        title of plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """

    if ax is None:
        _, ax = pu.make_fig()

    if trial_subset:
        data = trials_df.query("pro_anti_block_type == @trial_subset").copy()
    else:
        data = trials_df.copy()

    sns.violinplot(
        data=data,
        x="date",
        y="give_delay_dur",
        ax=ax,
        color="lightslategray",
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.5)
    _ = ax.set(title=title, xlabel="", ylabel="Give Del Dur [s]")

    return None


def plot_give_delay_dur_days_line(
    trials_df,
    ax=None,
    trial_subset="anti",
    title="",
    xaxis_label=False,
    aesthetics=True,
):
    """
    Plot the distribution of pre-give delay durations
    across days.

    params
    ------

    trials_df: pd.DataFrame
        trials dataframe with columns `pro_anti_block_type`,
        `date`, and `give_delay_dur` with trials as row index
    ax : matplotlib.axes.Axes, de
        axes to plot on
    title : str, (default=None)
        title of plot
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """

    if ax is None:
        _, ax = pu.make_fig()

    if trial_subset:
        data = trials_df.query("pro_anti_block_type == @trial_subset").copy()
    else:
        data = trials_df.copy()

    sns.lineplot(
        data=data,
        x="date",
        y="give_delay_dur",
        ax=ax,
        marker="o",
        color="lightslategray",
    )

    # aesthetics
    if aesthetics:
        pu.set_date_x_ticks(ax, xaxis_label)
        _ = ax.set(ylim=(0, None))
        ax.grid(alpha=0.5)
    _ = ax.set(title=title, xlabel="", ylabel="Give Del Dur [s]")

    return None


def plot_give_use_rate_days(
    trials_df,
    ax=None,
    trial_subset="anti",
    title="",
    rotate_x_labels=False,
):
    """
    Function to plot the give use rate across days as measured by the
    fraction of trials in which the give was delivered (ie no answer in the
    give delay period).

    params
    ------
    trials_df : pd.DataFrame
        trials dataframe with columns `pro_anti_block_type`,
        `date`, and `give_use` with trials as row index
    ax : matplotlib.axes.Axes, de
        axes to plot on
    trial_subset : str (default='anti')
        whether to plot the pro or anti trials
    title : str, (default=None)
        title of plot
    rotate_x_labels : bool (optional, default = False)
        whether to rotate the x-axis labels or not
    """
    if trial_subset:
        data = trials_df.query("pro_anti_block_type == @trial_subset").copy()
    else:
        data = trials_df.copy()

    if ax is None:
        _, ax = pu.make_fig()

    sns.lineplot(
        data=data.query("violations != 1"),
        x="date",
        y="give_use",
        marker="o",
        color="gold",
        ax=ax,
    )

    # plot alpha minus (used for adaptive threshold)
    sns.lineplot(
        data=data,
        x="date",
        y="give_del_adagrow_alpha_minus",
        marker="x",
        color="k",
        ax=ax,
        label="Alpha Minus",
    )

    # aesthetics
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.5)
    _ = ax.set(title=title, xlabel="", ylabel="Give Delivered Frac", ylim=(0, 1))

    return None
