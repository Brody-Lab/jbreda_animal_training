"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting functions for looking at training 
performance, water and mass metrics across days
"""

import seaborn as sns
from plotting_utils import *

# TODO move plots from `DMS_multiday_plots` to here and update df to be t_df
#######################
###  SUMMARY PLOTS  ###
#######################


def plot_multiday_summary(animal_id, d_df):
    """
    Plot the summary of the animal's performance over the
    date range in days_df

    params
    ------
    animal_id : str
        animal id to plot, e.g. "R610"
    d_df : pd.DataFrame
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

    animal_df = d_df.query("animal_id == @animal_id")

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


def plot_trials(d_df, ax, title="", legend=False, xaxis_label=True):
    """
    Plot the number of trials completed and trial rate over
    date range in d_d_df

    params
    ------
    d_df : pd.DataFrame
        days dataframe with columns `date`, `n_done_trials`, `trial_rate` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    """
    trial_melt = d_df.melt(
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


def plot_mass(d_df, ax, title="", xaxis_label=True):
    """
    Plot the mass of the animal over date range in d_df

    params
    ------
    d_df : pd.DataFrame
        days dataframe with columns `date`, `mass` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    """

    sns.lineplot(data=d_df, x="date", y="mass", marker="o", color="k", ax=ax)

    # aethetics
    set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    ax.set(ylabel="Mass [g]", xlabel="", title=title)

    return None


def plot_water_restriction(d_df, ax, title="", legend=True, xaxis_label=True):
    """
    Plot the rig, pub and restriction target volume over date
    range in d_df

    params
    ------
    d_df : pd.DataFrame
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
    d_df[columns_to_plot].plot(
        x="date",
        kind="bar",
        stacked=True,
        color=["blue", "cyan"],
        ax=ax,
    )

    # iterate over dates to plot volume target black line
    for i, row in d_df.reset_index().iterrows():
        ax.hlines(y=row["volume_target"], xmin=i - 0.35, xmax=i + 0.35, color="black")

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel="Volume [mL]")

    return None


def plot_rig_tech(d_df, ax, title="", legend=False, xaxis_label=True):
    """
    Plot the tech and rig id over date range in d_df

    params
    ------
    d_df : pd.DataFrame
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
    sns.lineplot(data=d_df, x="date", y="rigid", marker="o", color="gray", ax=ax)
    sns.lineplot(data=d_df, x="date", y="tech", marker="o", color="purple", ax=ax)

    set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(ylabel="Tech || Rig", xlabel="", title=title)
    ax.grid()

    return None


def plot_performance(d_df, ax, title="", legend=True, xaxis_label=True):
    """
    Plot the hit and violation rate over date range in d_df

    params
    ------
    d_df : pd.DataFrame
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
        data=d_df,
        x="date",
        y="hit_rate",
        marker="o",
        color="darkgreen",
        label="hit",
        ax=ax,
    )
    sns.lineplot(
        data=d_df,
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
    ax.set(ylim=(0, 1), ylabel="Performance", xlabel="", title=title)

    return None


def plot_side_bias(d_df, ax, title="", xaxis_label=True):
    """
    Plot the side bias over date range in d_df

    params
    ------
    d_df : pd.DataFrame
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
        data=d_df,
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
