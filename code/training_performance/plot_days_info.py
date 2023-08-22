"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting functions for looking at training 
performance, water and mass metrics across days
"""

import seaborn as sns
import pandas as pd
import numpy as np
import plot_utils as pu  # TODO make this a pu import
import matplotlib.pyplot as plt

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
    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
    _ = ax.set(ylabel="Count || Per Hr", xlabel="", title=title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)

    return None


### STAGE ###
def plot_stage(trials_df, ax, title="", xaxis_label=True, **kwargs):
    """
    TODO
    """
    sns.lineplot(
        data=trials_df.groupby("date").stage.mean(),
        drawstyle="steps-post",
        ax=ax,
        marker="o",
        **kwargs,
    )

    # aesthetics
    max_stage = int(trials_df.stage.max())
    pu.set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    ax.set(
        ylabel="Stage #",
        title=title,
        ylim=(0, max_stage + 1),
        yticks=range(max_stage + 1),
    )


### MASS ###
def plot_mass(days_df, ax, title="", xaxis_label=True):
    """
    Plot the mass of the animal over date range in days_df. If it's a mouse,
    will also plot mass relative to baseline as a percent.

    params
    ------
    days_df : pd.DataFrame
        days dataframe with columns `date`, `mass` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    """

    baseline_mass = get_baseline_mass(days_df)

    if baseline_mass is np.nan:
        plot_raw_mass(days_df, ax, title=title, xaxis_label=xaxis_label)
    else:
        plot_raw_and_relative_mass(
            days_df, baseline_mass, ax, title=title, xaxis_label=xaxis_label
        )

    return None


def get_baseline_mass(days_df):
    """
    Deterrmine if animal is a mouse or rat, so baseline mass plot
    can be determined. This assumes you have an "ANIMALS_TABLE",
    which is located in the data directory. I manually enter high-level
    animal info here that's not easy to access/store on datajoint.

    params
    ------
    days_df : pd.DataFrame
        days dataframe with `animal_id` column
    """

    from dir_utils import ANIMAL_TABLE_PATH

    animal_id = days_df.animal_id.unique()[0]
    animal_table = pd.read_excel(ANIMAL_TABLE_PATH)
    species = animal_table.query("animal_id == @animal_id").species.iloc[0]

    if species == "mouse":
        baseline_mass = animal_table.query(
            "animal_id == @animal_id"
        ).baseline_mass.iloc[0]
    else:
        baseline_mass = np.nan

    return baseline_mass


def plot_raw_mass(days_df, ax, title="", xaxis_label=True):
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
    """
    sns.lineplot(data=days_df, x="date", y="mass", marker="o", color="k", ax=ax)

    # aethetics
    pu.set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    ax.set(ylabel="Mass [g]", xlabel="", title=title)

    return None


def plot_raw_and_relative_mass(days_df, baseline_mass, ax, title="", xaxis_label=True):
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
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
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
    ax_r.set(ylabel="Relative Mass [%]")
    ax.grid()
    pu.set_date_x_ticks(ax, xaxis_label)

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
    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
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

    pu.set_date_x_ticks(ax, xaxis_label)
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
    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
    ax.grid(alpha=0.5)
    ax.set(ylim=(0, 1), ylabel="Perf Rate", xlabel="", title=title)

    return None


def plot_performance_bars(
    trials_df, ax, plot_type="counts", title="", legend=False, xaxis_label=True
):
    """
    Plot the count or rate of results in a stacked bar over date
    range in df

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and `result`
        with trials as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str, (default = "")
        title of plot
    legend : bool, (default = True)
        whether to include legend or not
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    """
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

    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel=f"Perf {plot_type}")

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
    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
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
    pu.set_date_x_ticks(ax, xaxis_label)
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
        marker="o",
        palette=["darkseagreen", "indianred"],
        ax=ax,
    )
    ax.axhline(0.5, color="k", linestyle="--", zorder=1)

    # aesthetics
    pu.set_date_x_ticks(ax, xaxis_label)
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
    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
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
        hue_order=pu.get_side_order(trials_df.first_spoke),
        marker="o",
        palette=pu.get_side_colors(trials_df.first_spoke),
        ax=ax,
    )

    # aesthetics
    pu.set_date_x_ticks(ax, xaxis_label)
    pu.set_legend(ax, legend)
    ax.set(ylabel="time to spoke [s]", xlabel="", title=title, ylim=(0))
    ax.grid(alpha=0.5)

    return None


### CPOKE ###


def plot_n_cpokes_and_multirate(trials_df, ax, title="", xaxis_label=True):
    """
    TODO
    """
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
    pu.set_date_x_ticks(ax, xaxis_label=xaxis_label)
    ax.grid()

    return None


def plot_cpoke_dur_timings_pregnp(trials_df, ax, title="", xaxis_label=True):
    """
    TODO
    """
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
    sns.lineplot(
        data=trials_df.query("violations == 1"),
        x="date",
        y="cpoke_dur",
        marker="o",
        ax=ax,
        color="orangered",
        label="Viol",
    )

    ax.set(ylabel="Duration [s]", xlabel="", title=title)
    ax.grid()
    pu.set_date_x_ticks(ax, xaxis_label)

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
        "post_dur",
    ]

    day_avgs = trials_df[columns_to_plot].groupby("date").mean().reset_index()
    day_avgs.insert(5, "s_b", day_avgs["stimulus_dur"])
    day_avgs.rename(columns={"stimulus_dur": "s_a"}, inplace=True)
    day_avgs.columns = day_avgs.columns.str.replace("_dur", "")
    day_avgs.plot(
        x="date",
        kind=kind,
        stacked=True,
        ax=ax,
        legend=legend,
        color=pu.trial_period_palette,
    )

    # aesthetics
    pu.set_date_x_ticks(ax, xaxis_label)
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
