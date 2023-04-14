import datajoint as dj
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import blob_transformation as bt
from pathlib import Path

SAVE_PATH = Path("X:\\jbreda\\animal_data\\training_data\\figures\\cohort_2_progress")

dj.blob.use_32bit_dims = True  # necessary for pd.blob read

bdata = dj.create_virtual_module("bdata", "bdata")
ratinfo = dj.create_virtual_module("intfo", "ratinfo")

##################################################
###               CALCULATIONS                 ###
##################################################


##############################
###        BHEAVIOR       ####
##############################
def calculate_daily_trial_rate(animal_id, date, units="hours"):
    """
    function that queries the sessions dj table to determine
    what the trial rate (in hours or minutes) was for an animal
    for a given day. Drops sessions with 1 or less trials from calc.

    params
    ------
    animal_id : str
        animal name, e.g. "R500"
    date : str or datetime
        date of interest in YYYY-MM-DD, e.g. "2023-04-12"
    units : str, "hours" (default) or "minutes"
        what units to return trial rate in
    """
    # fetch data
    query_keys = {
        "ratname": animal_id,
        "sessiondate": date,
    }  # specific to Sessions table
    n_done_trials, start_times, end_times = (bdata.Sessions & query_keys).fetch(
        "n_done_trials", "starttime", "endtime"
    )

    # convert to rate
    time_conversion = 1 / 3600 if units == "hours" else 1 / 60
    daily_train_time_seconds = (
        end_times[n_done_trials > 1] - start_times[n_done_trials > 1]
    )[0].seconds
    daily_trial_rate = np.sum(n_done_trials[n_done_trials > 1]) / (
        daily_train_time_seconds * time_conversion
    )

    return np.round(daily_trial_rate, decimals=2)

    ##############################
    ###      WATER/MASS       ####
    ##############################


def fetch_daily_water_target(animal_id, date, verbose=False):
    """
    Function for getting an animals water volume target on
    a specific date
    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04
    verbose : bool
        if you want to print restriction information
    returns
    ------
    volume_target : float
        water restriction target in mL
    """

    percent_target = fetch_daily_restriction_target(animal_id, date)
    mass = fetch_daily_mass(animal_id, date)

    # sometimes the pub isn't run- let's assume the minimum value
    if percent_target == 0:
        percent_target = 4
        note = "Note set to 0 but assumed 4."
    else:
        note = ""

    volume_target = np.round((percent_target / 100) * mass, 2)

    if verbose:
        print(
            f"""On {date} {animal_id} is restricted to:
        {percent_target}% of body weight or {volume_target} mL
        {note}
        """
        )

    return volume_target


def fetch_daily_restriction_target(animal_id, date):
    """
    Function for getting an animals water
    target for a specific date
    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04
    returns
    ------
    percent_target : float
        water restriction target in terms of percentage of body weight
    note
    ----
    You can also fetch this code from the registry, but it's not
    specific to the date. See code below.
    ```
    # fetch from comments section e.g. 'Mouse Water Pub 4'
    r500_registry = (ratinfo.Rats & 'ratname = "R501"').fetch1()
    comments = r500_registry['comments']
    #split after the word pub,only allow one split and
    # grab whatever follows the split
    target = float(comments.split('Pub',1)[1])
    ```
    """

    Water_keys = {"rat": animal_id, "date": date}

    # can't do fetch1 with this because water table
    # has a 0 entry and actual entry for every day
    # I'm taking the max to get around this
    # this needs to be address w/ DJ people
    percent_target = float((ratinfo.Water & Water_keys).fetch("percent_target").max())

    return percent_target


def fetch_daily_mass(animal_id, date):
    """
    Function for getting an animals mass on
    a specific date
    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04
    returns
    ------
    mass : float
        weight in grams on date
    """

    Mass_keys = {"ratname": animal_id, "date": date}
    mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))

    return mass


##################################################
###                  PLOTS                     ###
##################################################
def make_daily_spoke_stage_plot(df, overwrite=False):
    """
    TODO
    final plot format TBD
    """
    for (date, animal_id), sub_df in df.groupby(["date", "animal_id"]):
        fig_name = f"{animal_id}_{date}_daily_spoke_stage.png"

        if not Path.exists(SAVE_PATH / fig_name) or overwrite:
            print(f"plotting for {animal_id}")

            layout = """
                AAAB
                CCCD
                EEE.
                FFF.
                GGG.
            """
            fig = plt.figure(constrained_layout=True, figsize=(15, 15))

            plt.suptitle(f"\n{animal_id} on {sub_df.date[0]}\n", fontweight="semibold")
            ax_dict = fig.subplot_mosaic(layout)  # ax to plot to

            plot_daily_results(sub_df, ax=ax_dict["A"], title="trial result")
            plot_daily_water(sub_df, ax_dict["B"], title="water")
            plot_daily_npokes(sub_df, ax_dict["C"], plot_stage_info=False)
            plot_pokes_hist(sub_df, ax_dict["D"], title="pokes summary")
            plot_daily_first_spoke(
                sub_df, ax_dict["E"], title="first poke time", plot_stage_info=True
            )
            plot_daily_trial_dur(sub_df, ax_dict["F"], title="trial dur")
            plot_daily_perfs(sub_df, ax_dict["G"], title="performance")

            # save out
            plt.savefig(SAVE_PATH / fig_name[:-4], bbox_inches="tight")
            plt.close("all")


def plot_daily_results(df, ax, title=""):
    """
    plot trial result across a single day

    params
    ------
    df : DataFrame
        full protocol df thats n_done_trials for a day in length
         with "trial" and "result",  columns
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    """
    sns.scatterplot(
        data=df,
        x="trial",
        y="result",
        ax=ax,
        hue=df["result"].astype("category"),
        hue_order=get_result_order(df.result),
        palette=get_result_colors(df.result),
    )

    _ = ax.set(ylim=(0, 7), ylabel="Results", xlabel="Trials", title=title)
    ax.legend(
        labels=get_result_labels(df.result),
        loc="best",
        frameon=False,
        borderaxespad=0,
    )


def plot_daily_trial_dur(df, ax, title=""):
    """
    plot trial duration and iti across a single day

    params
    ------
    df : DataFrame
        full protocol df thats n_done_trials for a day in length
         with "trial", "trial_dur", "inter_trial_dur" columns
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    """

    durs_df = pd.melt(
        df, id_vars=["trial"], value_vars=["trial_dur", "inter_trial_dur"]
    )
    sns.lineplot(
        data=durs_df,
        x="trial",
        y="value",
        palette=["cornflowerblue", "gray"],
        hue="variable",
        ax=ax,
    )
    ax.axhline(df.trial_dur.mean(), color="cornflowerblue", linestyle="--", zorder=1)
    _ = ax.set(ylabel="Duration [s]", xlabel="Trials", title=title)
    ax.legend(loc="best", borderaxespad=0, frameon=False)


def plot_daily_npokes(df, ax, title="", plot_stage_info=True):
    """
    plot npokes (l,c,r) across a single day

    params
    ------
    df : DataFrame
        full protocol df thats n_done_trials for a day in length
         with "trial", "n_lpokes", "n_cpokes", "n_lpokes" columns
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    plot_stage_info : bool (default = True)
        whether or not add active stage number information,
        requires "stage" column
    """

    # make sub df
    pokes_df = pd.melt(
        df, id_vars=["trial"], value_vars=["n_lpokes", "n_cpokes", "n_rpokes"]
    )

    # plot pokes over trials
    sns.lineplot(
        data=pokes_df,
        x="trial",
        y="value",
        palette=["darkseagreen", "khaki", "indianred"],
        hue="variable",
        ax=ax,
    )

    if plot_stage_info:
        plot_daily_stage_info(df, pokes_df, ax)

    _ = ax.set(ylabel="N pokes", xlabel="Trials", title=title)
    ax.legend(loc="best", frameon=False, borderaxespad=0)


def plot_daily_first_spoke(df, ax, title="", plot_stage_info=False):
    """
    plot time to first spoke (l and/or r) across a single day

    params
    ------
    df : DataFrame
        full protocol df thats n_done_trials for a day in length
         with "trial", "first_spoke", "first_lpoke", "first_rpoke"
         columns
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot
    plot_stage_info : bool (default = False)
        whether or not add active stage number information,
        requires "stage" column
    """
    first_poke_df = pd.melt(
        df, id_vars=["trial", "first_spoke"], value_vars=["first_lpoke", "first_rpoke"]
    )
    df["was_no_answer"] = df["result"].apply(lambda row: True if row == 6.0 else np.nan)

    sns.scatterplot(
        data=first_poke_df,
        x="trial",
        y="value",
        ax=ax,
        hue="first_spoke",
        palette=get_poke_colors(first_poke_df.first_spoke),
    )
    sns.scatterplot(
        data=df, x="trial", y="was_no_answer", ax=ax, marker="s", color="black"
    )

    if plot_stage_info:
        plot_daily_stage_info(df, first_poke_df, ax)

    _ = ax.set(ylabel="Time to spoke [s]", xlabel="Trials", title=title)
    ax.legend(
        labels=get_poke_labels(first_poke_df.first_spoke),
        loc="best",
        frameon=False,
        borderaxespad=0,
    )


def plot_daily_stage_info(df, plot_df, ax):
    """
    plotting a bar indicating stage at top of plot

    params
    ------
    df : DataFrame
        full protocol df thats n_done_trials for a day in length
        with "stage", columns
    plot_df : DataFrame
        df that is actually being used for the plot (might be df,
        might also be a melted or grouped version of df)
    ax : matplotlib.axes
        axis to plot to
    """
    gray_palette = sns.color_palette("gray")
    for s in df.stage.unique():
        # calculate trial start and stop numbers for a given stage
        bounds = df[["trial", "stage"]].groupby("stage").agg(["min", "max"]).loc[s]
        ax.axvspan(
            xmin=bounds.min(),
            xmax=bounds.max() + 1,
            ymin=0.8,  # relative to plot (0,1)
            alpha=0.3,
            color=gray_palette[int(s - 1)],
        )
        # add text label
        ax.text(x=bounds.mean(), y=(plot_df.value.max() * 0.9), s=f"stage {int(s)}")


def plot_daily_perfs(df, ax, title=""):
    """
    TODO
    """
    perf_rates_df = pd.melt(
        df, id_vars=["trial"], value_vars=["violation_rate", "error_rate", "hit_rate"]
    )
    sns.lineplot(data=perf_rates_df, x="trial", y="value", hue="variable")
    _ = ax.set(ylabel="Rate", xlabel="Trials", title=title)
    ax.legend(loc="best", frameon=False, borderaxespad=0)


def plot_pokes_hist(df, ax, title=""):
    """
    TODO
    """
    pokes_df = pd.melt(
        df, id_vars=["trial"], value_vars=["n_lpokes", "n_cpokes", "n_rpokes"]
    )

    sns.histplot(
        data=pokes_df,
        x="value",
        palette=["darkseagreen", "khaki", "indianred"],
        hue="variable",
        element="step",
        ax=ax,
        legend=False,
    )

    ax.set(title=title)
    ##############################
    ####      WATER/MASS      ####
    ##############################


def plot_daily_water(df, ax, title=""):
    """
    Quick function for plotting water consumed in rig or pub for a day
    and marking the target with a horizontal line
    """
    # TODO add option for plotting a specific date

    animal_id = df["animal_id"][0]
    date = df["date"][0]

    Water_keys = {"rat": animal_id, "date": date}  # specific to Water table
    pub_volume = float((ratinfo.Water & Water_keys).fetch("volume").max())
    rig_volume = df.water_delivered.sum() / 1000  # convert to mL
    volume_target = fetch_daily_water_target(animal_id, date, verbose=False)

    df = pd.DataFrame(
        {"date": [date], "rig_volume": [rig_volume], "pub_volume": [pub_volume]}
    )

    # plot
    df.set_index("date").plot(kind="bar", stacked=True, color=["blue", "cyan"], ax=ax)
    ax.axhline(y=volume_target, xmin=0.2, xmax=0.8, color="black")

    # legend
    order = [1, 0]
    handles, _ = ax.get_legend_handles_labels()
    labels = ["rig", "pub"]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc=(0.9, 0.75))

    # aesthetics
    ax.set_xticks([])
    # _ = ax.set_xticklabels(ax.get_xticks(), rotation=45)
    _ = ax.set(xlabel="", ylabel="volume (mL)", title=title)
    sns.despine()


##################################################
###               PLOTS AESTHETICS             ###
##################################################
POKE_MAP = {
    "l": {"label": "left", "color": "darkseagreen"},
    "c": {"label": "center", "color": "khaki"},
    "r": {"label": "right", "color": "indianred"},
    "n": {"label": "", "color": "white"},
}

RESULT_MAP = {
    1: {"label": "hit", "color": "yellowgreen"},
    2: {"label": "error", "color": "maroon"},
    3: {"label": "viol", "color": "organgered"},
    4: {"label": "terr", "color": "lightcoral"},
    5: {"label": "crash", "color": "cyan"},
    6: {"label": "noans", "color": "black"},
}


def get_poke_colors(poke_column):
    pokes = poke_column.unique()  # any colum with 'l', 'r', or 'c'
    colors = [POKE_MAP[poke]["color"] for poke in pokes]
    return colors


def get_poke_labels(poke_column):
    pokes = poke_column.unique()  # any colum with 'l', 'r', or 'c'
    colors = [POKE_MAP[poke]["label"] for poke in pokes]
    return colors


def get_result_labels(result_column):
    results = result_column.unique()
    labels = [RESULT_MAP[res]["label"] for res in results]
    return labels


def get_result_colors(result_column):
    # sorted to match hue_order
    results = result_column.sort_values().unique()
    colors = [RESULT_MAP[res]["color"] for res in results]
    return colors


def get_result_order(result_column):
    return result_column.sort_values().unique()


def get_poke_pallete(poke_column):
    pokes = poke_column.unique()

    poke_map = {}
    if "l" in pokes and "r" in pokes and "c" in pokes:
        palette = ["darkseagreen", "khaki", "indianred"]
    elif "l" in pokes and "r" in pokes:
        palette = ["darkseagreen", "indianred"]
    elif "l" in pokes:
        palette = ["darkseagreen"]
    elif "r" in pokes:
        palette = ["indianred"]
    elif "c" in pokes:
        palette = ["khaki"]

    return palette


# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : Dict[str, Axes]
        Mapping between the title / label and the Axes.

    fontsize : int, optional
        How big the label should be
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


# TODO write in a get_pokes_order to keep l first awalsy
