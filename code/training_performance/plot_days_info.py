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
def plot_stage(
    trials_df, ax, title="", group="date", xaxis_label=True, aesthetics=True, **kwargs
):
    """
    Having the group variable allows you to group by other
    things like "start_date" if you want to try and make a plot
    with multiple animals that started at different times.
    """
    sns.lineplot(
        data=trials_df.groupby(group).stage.mean(),
        drawstyle="steps-post",
        ax=ax,
        marker="o",
        **kwargs,
    )
    if aesthetics:
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
    ax.set(title=title, xlabel="", ylabel="Prob", ylim=(0, 1))
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


### GIVE ###


def plot_performance_by_give(
    trials_df,
    ax,
    title="",
    group="date",
    xaxis_label=True,
    legend=True,
    aesthetics=True,
):
    """
    generate a plot of hit rate for non-give trials

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `hits` and
        `give_type_imp` with trials as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str, (default = "")
        title of plot
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
    if aesthetics:
        _ = ax.set(ylabel="Proportion", xlabel="", title=title, ylim=(0, 1))
        if legend:
            ax.legend(loc="lower left")
        else:
            ax.legend().remove()
        pu.set_date_x_ticks(ax, xaxis_label)
    else:
        _ = ax.set(title=title, ylabel="Performance", xlabel="Days in Stage 10")
    ax.grid(alpha=0.5)
    return None


def plot_give_info_days(
    trials_df, ax, title="", aesthetics=True, xaxis_label=True, legend=False
):
    """
    Plot the give information across days.

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `give_type_imp` with trials as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str, (default = "")
        title of plot
    aesthetics : bool (optional, default = True)
        used to toggle xaxis label when subplotting
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    legend : bool (optional, default = True)
        whether to include the legend or not
    """
    # make the names shorter for plotting
    data = trials_df[["date", "give_type_imp"]].copy()
    mapping = {"water_and_light": "w + l", "water": "w", "light": "l", "none": "n"}
    data.give_type_imp = data.give_type_imp.replace(mapping)

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
    _ = ax.set(title=title, ylabel="", xlabel="")
    if aesthetics:
        pu.set_legend(ax, legend)
        pu.set_date_x_ticks(ax, xaxis_label)

    return None


### SOUNDS ###


def plot_sounds_info(trials_df, ax, title="", xaxis_label=True):
    """
    Plot the sound volume and duration info across days

    params
    ------
    trials_df : pandas.DataFrame
        trials dataframe with columns `date` and
        `volume_multiplier` `stimulus_dur` with trials as
        row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str, (default = "")
        title of plot
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
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
    _ = ax.set(title=title, ylabel="Sound Variable Value", xlabel="")
    pu.set_date_x_ticks(ax, True)
    ax.legend(loc="center left")

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


## PRO ANTI ##


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


def plot_stim_performance_by_pro_anti(
    trials_df, ax, x_var="date", title="", xaxis_label=True, aesthetics=True
):
    """
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
    trials_df, ax=None, x_var="date", title="", xaxis_label=True, aesthetics=True
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
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
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
    if aesthetics:
        _ = ax.set(
            ylabel="N Blocks",
            xlabel="",
            title=title,
            ylim=(0, trials_df.n_blocks.max() + 1),
        )
        pu.set_date_x_ticks(ax, xaxis_label)
    else:
        ax.set(ylim=(0, trials_df.n_blocks.max() + 1))
    ax.grid(alpha=0.5)

    return None


def plot_block_switch_thresholds(trials_df, ax=None, title="", xaxis_label=True):
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
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    legend : bool (optional, default = True)
        whether to include the legend or not


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
    pu.set_date_x_ticks(ax, xaxis_label)

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
    trials_df, ax=None, x_var="date", title="", xaxis_label=True, aesthetics=True
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
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
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
    if aesthetics:
        _ = ax.set(
            ylabel="Min. Block Size",
            xlabel="",
            title=title,
            ylim=(0, trials_df.block_size.max() + 1),
        )
        pu.set_date_x_ticks(ax, xaxis_label)
    else:
        ax.set(ylim=(0, trials_df.block_size.max() + 1))
    ax.grid(alpha=0.5)

    return None


def plot_give_type_and_block_switch_days(
    trials_df, ax=None, title="", xaxis_label=True, legend=False
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
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure
    legend : bool (optional, default = True)
        whether to include the legend or not
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
    pu.set_date_x_ticks(ax, xaxis_label)

    return None


def plot_block_switch_params(trials_df, ax=None, title="", xaxis_label=False):
    """
    TODO

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
    xaxis_label : bool (optional, default = True)
        whether to include the xaxis label or not, this is useful when
        plotting multiple plots on the same figure

    """
    if ax is None:
        _, ax = pu.make_fig()

    plot_min_block_size(trials_df, ax=ax, xaxis_label=xaxis_label)
    ax.set_label("Min Block Size")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    plot_block_switch_thresholds(trials_df, ax=ax2, xaxis_label=xaxis_label)

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

    sns.violinplot(
        data=data,
        x="date",
        y="give_delay_dur",
        ax=ax,
        color="lightslategray",
    )

    # aesthetics
    if aesthetics:
        pu.set_date_x_ticks(ax, xaxis_label)
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
    xaxis_label=False,
    aesthetics=True,
):
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
    if aesthetics:
        pu.set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    _ = ax.set(title=title, xlabel="", ylabel="Give Delivered Frac", ylim=(0, 1))

    return None
