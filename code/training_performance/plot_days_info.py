"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting functions for looking at training 
performance, water and mass metrics across days
"""

import seaborn as sns
import pandas as pd
from plotting_utils import *

# TODO move plots from `DMS_multiday_plots` to here and update df to be trials_df
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
    plot_trials(animal_df, ax_dict["A"], title="Trials", legend=True, xaxis_label=False)
    plot_performance(animal_df, ax_dict["C"], title="Performance", xaxis_label=False)
    plot_side_bias(animal_df, ax_dict["E"], title="Side Bias", xaxis_label=True)

    # right column
    plot_mass(animal_df, ax_dict["B"], title="Mass", xaxis_label=False)
    plot_water_restriction(
        animal_df, ax_dict["D"], title="Water", legend=False, xaxis_label=False
    )
    plot_rig_tech(animal_df, ax_dict["F"], title="Rig/Tech", xaxis_label=True)

    return None


######################
###  SINGLE PLOTS  ###
######################


### TRIALS ###
def plot_trials(days_df, ax, title="", legend=False, xaxis_label=True):
    """
    Plot the number of trials completed and trial rate over
    date range in d_days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `n_done_trials`, `trial_rate` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    """
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

    # aethetics
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    _ = ax.set(ylabel="Count || Per Hr", xlabel="", title=title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)

    return None


### STAGE ###
def plot_stage(trials_df, ax, max_stage=8, title="", xaxis_label=True, **kwargs):
    """
    TODO
    """
    sns.lineplot(
        data=trials_df.groupby("date").stage.mean(),
        drawstyle="steps-post",
        ax=ax,
        **kwargs,
    )

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    _ = plt.yticks(range(1, max_stage, 1))
    _ = ax.set(ylabel="Stage #", title=title)


### MASS ###
def plot_mass(days_df, ax, title="", xaxis_label=True):
    """
    Plot the mass of the animal over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `mass` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    """

    sns.lineplot(data=days_df, x="date", y="mass", marker="o", color="k", ax=ax)

    # aethetics
    set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    ax.set(ylabel="Mass [g]", xlabel="", title=title)

    return None


### WATER ###
def plot_water_restriction(days_df, ax, title="", legend=True, xaxis_label=True):
    """
    Plot the rig, pub and restriction target volume over date
    range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `rig_volume`, `pub_volume`
        and `volume_target` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """

    # stacked bar chart only works with df.plot (not seaborn)
    columns_to_plot = ["date", "rig_volume", "pub_volume"]
    days_df[columns_to_plot].plot(
        x="date",
        kind="bar",
        stacked=True,
        color=["blue", "cyan"],
        ax=ax,
    )

    # iterate over dates to plot volume target black line
    for i, row in days_df.reset_index().iterrows():
        ax.hlines(y=row["volume_target"], xmin=i - 0.35, xmax=i + 0.35, color="black")

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel="Volume [mL]")

    return None


### RIG/TECH ###
def plot_rig_tech(days_df, ax, title="", legend=False, xaxis_label=True):
    """
    Plot the tech and rig id over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `rigid` and `tech` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """
    sns.lineplot(data=days_df, x="date", y="rigid", marker="o", color="gray", ax=ax)
    sns.lineplot(data=days_df, x="date", y="tech", marker="o", color="purple", ax=ax)

    set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(ylabel="Tech || Rig", xlabel="", title=title)
    ax.grid()

    return None


### PERFORMANCE ###
def plot_performance(days_df, ax, title="", legend=True, xaxis_label=True):
    """
    Plot the hit and violation rate over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `hit_rate` and `viol_rate` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """
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
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.grid(alpha=0.5)
    ax.set(ylim=(0, 1), ylabel="Perf Rate", xlabel="", title=title)

    return None


def plot_performance_bars(
    trial_df, ax, normalize=False, title="", legend=False, xaxis_label=True
):
    """
    TODO
    """
    ylabel = "eates" if normalize else "counts"
    perf_df = (
        trial_df.groupby(["date"]).result.value_counts(normalize=normalize).unstack()
    )

    perf_df.plot(kind="bar", stacked=True, ax=ax)

    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel=f"Perf {ylabel}")

    return None


def plot_performance_w_error(trials_df, ax, title="", legend=False, xaxis_label=True):
    """
    Plot hit, error and violation rate over date range in trials df

    params
    ------
    trials_df :

    TODO

    """
    perf_rates_df = pd.melt(
        trials_df,
        id_vars=["date"],
        value_vars=["violation_rate", "error_rate", "hit_rate"],
    )

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
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.grid(alpha=0.5)
    _ = ax.set(ylabel="Perf Rate", xlabel="", title=title)
    ax.set(ylim=(0, 1))

    return None


### SIDE BIAS ###
def plot_side_bias(days_df, ax, title="", xaxis_label=True):
    """
    Plot the side bias over date range in days_df

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date` and `side_bias` with
        dates as row index, positive values = right bias
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """
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
    set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    ax.set(ylim=(-1, 1), ylabel="< - Left | Right ->", xlabel="", title=title)

    return None


def plot_antibias_probs(trials_df, ax, title="", legend=True, xaxis_label=True):
    """
    TODO
    """
    ab_melt = trials_df.melt(
        id_vars=["date"], value_vars=["ab_l_prob", "ab_r_prob"], value_name="antibias"
    )

    sns.lineplot(
        data=ab_melt,
        x="date",
        y="antibias",
        hue="variable",
        palette=["darkseagreen", "indianred"],
        ax=ax,
    )
    ax.axhline(0.5, color="k", linestyle="--", zorder=1)

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    ax.set(title=title, xlabel="", ylabel="Prob")
    ax.legend(frameon=False, borderaxespad=0)

    return None


def plot_sidebias_params(trials_df, ax, title="", legend=False, xaxis_label=True):
    """
    TODO
    """

    sidebias_melt = trials_df.melt(
        id_vars=["date"],
        value_name="sb_vars",
        value_vars=["l_water_vol", "r_water_vol", "ab_beta"],
    )
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
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.grid(axis="y")
    _ = ax.set(title=title, xlabel="", ylabel="Value")

    return None


### SIDE POKE ###
def plot_time_to_spoke(trials_df, ax, title="", legend=True, xaxis_label=True):
    sns.lineplot(
        data=trials_df,
        x="date",
        y="min_time_to_spoke",
        hue="first_spoke",
        palette=["darkseagreen", "indianred", "white"],
        ax=ax,
    )

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.set(ylabel="time to spoke [s]", xlabel="", title=title, ylim=(0))
    ax.grid(alpha=0.5)

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
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    _ = ax.set(title=title, xlabel="", ylabel="Delay Dur [s]")

    return None


def plot_avg_delay(trials_df, ax, title="", xaxis_label=True):
    """
    TODO
    """
    sns.lineplot(data=trials_df, x="date", y="delay_dur", marker="o", ax=ax)

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(title=title, xlabel="", ylabel="Avg Delay [s]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)

    return None


### TIMING ###
def plot_trial_structure(
    trials_df, ax, kind="bar", title="", legend=True, xaxis_label=True
):
    """
    TODO
    """
    columns_to_plot = [
        "date",
        "settling_in_dur",
        "adj_pre_dur",
        "stimulus_dur",
        "delay_dur",
    ]

    day_avgs = trials_df[columns_to_plot].groupby("date").mean().reset_index()

    day_avgs.plot(x="date", kind=kind, stacked=True, ax=ax, legend=legend)

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(title=title, xlabel="", ylabel="Trial Timing [s]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5, axis="y")

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
    set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(title=title, xlabel="", ylabel="Trial Timing [s]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5, axis="y")

    return None
