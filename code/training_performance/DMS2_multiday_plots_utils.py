import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


####################
###    TRIALS    ###
####################
def plot_multianimal_trials(df, ax, title=""):
    """
    TODO
    """
    sns.lineplot(
        data=df.groupby(["animal_id", "date"]).max(),
        x="date",
        y="trial",
        hue="animal_id",
        palette="Oranges",
        ax=ax,
    )

    ax.tick_params(axis="x", labelrotation=45)
    _ = ax.set(ylabel="trials per session", title="N Trials")
    sns.despine()

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0, frameon=False)


def plot_multiday_trials(df, ax, title="", legend=False):
    """
    Plot the number of trials completed per day and the trial rate

    params
    ------
    df : pd.DataFrame
        dataframe with columns `date`, `n_done_trials`, `trial_rate` with
        date as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    """

    trial_melt = df.melt(
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

    # aesthetics
    set_date_x_ticks(ax, len(df.date.unique()))
    set_legend(ax, legend)
    _ = ax.set(ylabel="Count || Per Hr", xlabel="", title=title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)


####################
### HIT/VIOL/ETC ###
####################
def plot_multiday_perfs(df, ax, title="", legend=False):
    perf_rates_df = pd.melt(
        df, id_vars=["date"], value_vars=["violation_rate", "error_rate", "hit_rate"]
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

    set_date_x_ticks(ax, len(df.date.unique()))
    set_legend(ax, legend)

    ax.grid(alpha=0.5)
    _ = ax.set(ylabel="Rate", xlabel="", title=title)
    ax.set(ylim=(0, 1))


####################
###    STAGE     ###
####################
def plot_stage(df, ax, **kwargs):
    sns.lineplot(
        data=df.groupby("date").stage.mean(), drawstyle="steps-post", ax=ax, **kwargs
    )

    _ = plt.xticks(rotation=45)
    _ = plt.yticks(np.arange(1, 6, 1))
    _ = ax.set(ylabel="stage number", title="")


####################
###    DELAY     ###
####################


def plot_multiday_delay_params(df, ax, title="", legend=False):
    sns.lineplot(data=df, x="date", y="delay_dur", legend=legend, ax=ax)

    delay_melt = df.melt(
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

    set_date_x_ticks(ax, len(df.date.unique()))
    set_legend(ax, legend)
    _ = ax.set(title=title, xlabel="", ylabel="Delay Dur [s]")


def plot_multiday_avg_delay(df, ax, title=""):
    sns.lineplot(data=df, x="date", y="delay_dur", marker="o", ax=ax)

    set_date_x_ticks(ax, len(df.date.unique()))
    _ = ax.set(title=title, xlabel="", ylabel="Avg Delay [s]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)


####################
### MASS & WATER ###
####################


def plot_multiday_mass(df, ax, title=""):
    """
    Plot the mass of the animal over date range in df

    params
    ------
    df : pd.DataFrame
        dataframe with columns `date`, `mass` with date as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    """

    sns.lineplot(data=df, x="date", y="mass", marker="o", color="k", ax=ax)

    # aesthetics
    set_date_x_ticks(ax, len(df.date.unique()))
    ax.grid(alpha=0.5)
    ax.set(ylabel="Mass [g]", xlabel="", title=title)


def plot_multiday_water_restriction(df, ax, title="", legend=True):
    """
    Plot the rig, pub and restriction target volume over data
    range in df

    params
    ------
    df : pd.DataFrame
        dataframe with columns `date`, `rig_volume`, `pub_volume`
        and `volume_target` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    """
    # stacked bar chart only works with df.plot (not seaborn)
    columns_to_plot = ["date", "rig_volume", "pub_volume"]
    df[columns_to_plot].plot(
        x="date", kind="bar", stacked=True, color=["blue", "cyan"], ax=ax
    )

    # iterate over dates to plot volume target black line
    for i in range(len(df)):
        ax.hlines(y=df.volume_target[i], xmin=i - 0.5, xmax=i + 0.5, color="black")

    # aesthetics
    set_date_x_ticks(ax)
    set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel="Volume [mL]")

    return None


####################
###   RIG/TECH   ###
####################
def plot_multiday_rig_tech(df, ax, title="", legend=False):
    """
    Plot the tech and rig id over data range in df

    params
    ------
    df : pd.DataFrame
        dataframe with columns `date`, `rigid` and `tech` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    """
    sns.lineplot(data=df, x="date", y="rigid", marker="o", ax=ax)
    sns.lineplot(data=df, x="date", y="tech", marker="o", ax=ax)

    set_date_x_ticks(ax)
    _ = ax.set(ylabel="Tech || Rig", xlabel="", title=title)
    ax.grid()

    return None


####################
###     BIAS     ###
####################
def calculate_bias_history(df):
    """
    Function to compute the bias side bias history
    over previous days for a given animal

    params
    ------
    df : DataFrame
        with trial by trial data and columns "date"
        and "side"

    returns
    -------
    bias_df : DataFrame
        to be used in plotting, side bias for each data
        in df. L = +1, R = -1
    """

    # calculate bias
    bias_df = pd.DataFrame(columns=["date", "bias"])

    for date, date_df in df.groupby("date"):
        side_perf = date_df.groupby("sides").hits.mean().reset_index()

        # always make sure left row is first
        side_perf = side_perf.sort_values(by="sides")

        # skip over days with only one side
        if len(side_perf) == 1:
            continue

        # left hits - right hits, 2 decimal points
        bias = round(side_perf.hits.iloc[0] - side_perf.hits.iloc[1], 2)

        bias_df = pd.concat(
            [bias_df, pd.DataFrame({"date": [date], "bias": [bias]})], ignore_index=True
        )

    return bias_df


def plot_multiday_side_bias(df, ax, **kwargs):
    bias_df = calculate_bias_history(df)
    # if having issues with time plots, try this
    # bias_df["date"] = bias_df["date"].astype(str)

    sns.lineplot(
        data=bias_df,
        x="date",
        y="bias",
        errorbar=None,
        marker="o",
        ax=ax,
        **kwargs,
    )
    ax.axhline(0, color="k", linestyle="--", zorder=1)

    set_date_x_ticks(ax, len(df.date.unique()))
    _ = ax.set(ylabel="<- R bias | L bias ->", title="Side Bias", ylim=[-1, 1])

    return None


def plot_multiday_water_vols(df, ax):
    water_melt = df.melt(
        id_vars=["date"],
        value_name="water_vol",
        value_vars=["l_water_vol", "r_water_vol"],
    )

    sns.barplot(
        data=water_melt,
        x="date",
        y="water_vol",
        hue="variable",
        palette=["teal", "purple"],
        alpha=0.5,
        ax=ax,
    )

    set_date_x_ticks(ax, len(df.date.unique()))
    _ = ax.set(title="", xlabel="", ylabel="Volume (uL)")
    ax.legend(frameon=False, borderaxespad=0)


def plot_multiday_antibias_probs(df, ax):
    ab_melt = df.melt(
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

    set_date_x_ticks(ax, len(df.date.unique()))
    ax.set(title="Antibias L/R probs", xlabel="", ylabel="Probability")
    ax.legend(frameon=False, borderaxespad=0)


def plot_multiday_antibias_beta(df, ax):
    """
    TODO add gridlines
    """

    sns.barplot(
        data=df, x="date", y="ab_beta", color="cornflowerblue", errorbar=None, ax=ax
    )
    set_date_x_ticks(ax, len(df.date.unique()))
    _ = ax.set(ylim=(-0.5, df.ab_beta.max() + 1), ylabel="Beta", xlabel="")


def plot_multiday_sidebias_params(df, ax, title="", legend=False):
    """
    TODO
    """
    sidebias_melt = df.melt(
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

    set_date_x_ticks(ax, len(df.date.unique()))
    set_legend(ax, legend)

    _ = ax.set(title=title, xlabel="", ylabel="Value")


def plot_multiday_time_to_spoke(df, ax, title="", legend=True):
    sns.lineplot(
        data=df,
        x="date",
        y="min_time_to_spoke",
        hue="first_spoke",
        palette=["darkseagreen", "indianred", "white"],
        ax=ax,
    )

    set_date_x_ticks(ax, len(df.date.unique()))
    set_legend(ax, legend)

    ax.set(ylabel="time to spoke [s]", xlabel="", title=title, ylim=(0))
    ax.grid(alpha=0.5)


##### UTILS ####


def create_figure(figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def set_date_x_ticks(ax, n_dates, n_plots=1):
    ticks = ax.get_xticks()

    # if n_dates > 25:
    #     # skip every other label
    #     ax.set_xticks(ticks[::2])  # skip every other label

    ax.set_xticks(ticks)

    ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=45)


def set_legend(ax, legend):
    if legend:
        ax.legend(frameon=False, borderaxespad=0)
    else:
        ax.get_legend().remove()
