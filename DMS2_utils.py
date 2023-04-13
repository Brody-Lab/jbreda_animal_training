import datajoint as dj
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import blob_transformation as bt


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


def plot_trial_and_iti_dur(df, ax, title=""):
    """
    plot trial duration and iti across a single day

    params
    ------
    df : DataFrame
        full protocol df thats n_done_trials for a day in length
         with "trials", "trial_dur", "inter_trial_dur" columns
    ax : matplotlib.axes
        axis to plot to
    title : str, (default = "")
        title of plot

    returns
    ------
    none
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
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)


def plot_npokes_across_trials(df, ax, title="", plot_stage_info=True):
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

    returns
    ------
    none
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
            ax.text(
                x=bounds.mean(), y=(pokes_df.value.max() * 0.9), s=f"stage {int(s)}"
            )

    _ = ax.set(ylabel="N pokes", xlabel="Trials", title=title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    ##############################
    ####      WATER/MASS      ####
    ##############################


def plot_daily_water(df, ax):
    """
    Quick function for plotting water consumed in rig or pub for a day
    and marking the target with a horizontal line
    """
    # TODO add option for plotting a specific date

    animal_id = df["animal_id"][0]
    date = df["date"][0].date()

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
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[i] for i in order], [labels[i] for i in order], loc=(0.9, 0.75))

    # aesthetics
    _ = plt.xticks(rotation=45)
    _ = ax.set(xlabel="", ylabel="volume (mL)", title=f"{animal_id} water info")
    sns.despine()


##################################################
###               PLOTS AESTHETICS             ###
##################################################


def get_poke_pallete(poke_column):
    pokes = poke_column.unique()
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
