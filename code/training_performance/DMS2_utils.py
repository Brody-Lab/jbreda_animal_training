import datajoint as dj
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import dj_utils as djut
from pathlib import Path
from datetime import timedelta

SAVE_PATH = Path("X:\\jbreda\\animal_data\\training_data\\figures\\cohort_2_progress")

dj.blob.use_32bit_dims = True  # necessary for pd.blob read

bdata = dj.create_virtual_module("bdata", "bdata")
ratinfo = dj.create_virtual_module("intfo", "ratinfo")


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
    # sometimes has a 0 entry and actual entry so
    # I'm taking the max to get around this
    # this needs to be address w/ DJ people
    percent_target = (ratinfo.Water & Water_keys).fetch("percent_target")

    if len(percent_target) == 0:
        percent_target = 4  # NOTE assumption made here to 4%- be careful!
    elif len(percent_target) > 1:
        percent_target = percent_target.max()

    return float(percent_target)


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
    try:
        mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))
    except:
        print(
            f"mass data not found for {animal_id} on {date},",
            f"using previous days mass",
        )
        prev_date = date - timedelta(days=1)
        Mass_keys = {"ratname": animal_id, "date": prev_date}
        mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))
    return mass


##################################################
###                  PLOTS                     ###
##################################################


def make_daily_stage_plots(df, overwrite=False):
    for (date, animal_id), sub_df in df.groupby(["date", "animal_id"]):
        if sub_df.stage.max() < 3:
            make_daily_spoke_stage_plot(
                sub_df, overwrite=overwrite, date=date, animal_id=animal_id
            )
        else:
            make_daily_stage_3_4_plot(
                sub_df, overwrite=overwrite, date=date, animal_id=animal_id
            )


def make_daily_stage_3_4_plot(df, overwrite=False, date=None, animal_id=None):
    # if we don't get date/animal id values assume we are working
    # with a groupby df and grab the first dow
    if date is None:
        date = df.date.iloc[0]
    if animal_id is None:
        animal_id = df.animal_id.iloc[0]

    fig_name = f"{animal_id}_{date}_daily_stage_3_4.png"

    if not Path.exists(SAVE_PATH / fig_name) or overwrite:
        print(f"plotting stage 3 / 4 plot {animal_id} on {date}")

        layout = """
            AAABCCC
            DDDEFFF
            GGGHIJK
            LLLMNNN
        """
        fig = plt.figure(constrained_layout=True, figsize=(20, 15))

        max_stage = df.stage.max()

        plt.suptitle(f"\n{animal_id} on {date}\n", fontweight="semibold")
        ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
        # identify_axes(ax_dict) # prints the letter for id

        ## ROW 1
        plot_daily_results(df, ax=ax_dict["A"], title="trial result")
        plot_daily_result_summary(df, ax=ax_dict["B"], title="frac result")
        plot_daily_perfs(df, ax_dict["C"], title="performance")

        ## ROW 2
        plot_daily_npokes(df, ax_dict["D"], plot_stage_info=False)
        plot_pokes_hist(df, ax_dict["E"], title="pokes summary")
        plot_daily_first_spoke(
            df, ax_dict["F"], title="first poke time", plot_stage_info=True
        )

        ## ROW 3
        # if max_stage == 3:
        #     plot_daily_delay_dur(df, ax_dict["G"], title="delay")
        #     plot_type = "early"
        plot_daily_viol_pre_go_durs(df, ax_dict["G"])

        plot_delay_poke_hist(df, ax_dict["H"])

        plot_daily_first_spoke_hist(df, ax_dict["I"])
        plot_daily_side_bias(df, ax=ax_dict["J"])
        plot_daily_side_counts(df, ax=ax_dict["K"])

        # ROW 4
        plot_daily_trial_dur(df, ax_dict["L"], title="trial dur")
        plot_daily_water(df, ax_dict["M"], title="water")
        # ANTIBIAS GOES HERE
        # plot_daily_trial_dur(df, ax_dict["N"], title="trial dur")

        # save out
        plt.savefig(SAVE_PATH / fig_name[:-4], bbox_inches="tight")
        plt.close("all")


def make_daily_spoke_stage_plot(df, overwrite=False, date=None, animal_id=None):
    # if we don't get date/animal id values assume we are working
    # with a groupby df and grab the first dow
    if date is None:
        date = df.date.iloc[0]
    if animal_id is None:
        animal_id = df.animal_id.iloc[0]

    fig_name = f"{animal_id}_{date}_daily_spoke_stage.png"

    if not Path.exists(SAVE_PATH / fig_name) or overwrite:
        print(f"plotting spoke plot {animal_id} on {date}")

        layout = """
            AAAB
            CCCD
            EEEF
            GGG.
            HHH.
        """
        fig = plt.figure(constrained_layout=True, figsize=(15, 15))

        plt.suptitle(f"\n{animal_id} on {date}\n", fontweight="semibold")
        ax_dict = fig.subplot_mosaic(layout)  # ax to plot to

        plot_daily_results(df, ax=ax_dict["A"], title="trial result")
        plot_daily_result_summary(df, ax=ax_dict["B"], title="frac result")
        plot_daily_npokes(df, ax_dict["C"], plot_stage_info=False)
        plot_pokes_hist(df, ax_dict["D"], title="pokes summary")
        plot_daily_first_spoke(
            df, ax_dict["E"], title="first poke time", plot_stage_info=True
        )
        plot_daily_water(df, ax_dict["F"], title="water")
        plot_daily_trial_dur(df, ax_dict["G"], title="trial dur")
        plot_daily_perfs(df, ax_dict["H"], title="performance")

        # save out
        plt.savefig(SAVE_PATH / fig_name[:-4], bbox_inches="tight")
        plt.close("all")


def plot_daily_results(df, ax, title=""):
    """
    DONE
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
        legend=False,
    )

    _ = ax.set(ylim=(0, 7), ylabel="Results", xlabel="Trials", title=title)


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
    DONE
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
    was_no_answer = df["result"].apply(lambda row: True if row == 6.0 else np.nan)

    sns.scatterplot(
        data=first_poke_df,
        x="trial",
        y="value",
        ax=ax,
        hue="first_spoke",
        palette=get_poke_colors(first_poke_df.first_spoke),
    )
    ax.scatter(x=df.trial, y=was_no_answer, marker="s", color="black")

    if plot_stage_info:
        plot_daily_stage_info(df, first_poke_df, ax)

    _ = ax.set(ylabel="Time to spoke [s]", xlabel="Trials", title=title)
    ax.legend(frameon=False, borderaxespad=0)


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
    sns.lineplot(
        data=perf_rates_df,
        x="trial",
        y="value",
        hue="variable",
        palette=["orangered", "maroon", "darkgreen"],
        ax=ax,
    )
    _ = ax.set(ylabel="Rate", xlabel="Trials", title=title, ylim=(0, 1))
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


def plot_daily_result_summary(df, ax, title=""):
    """
    DONE
    """
    res_summary = df.result.value_counts(normalize=True).sort_index(ascending=True)
    res_summary.plot(kind="bar", color=get_result_colors(df.result), ax=ax)

    _ = ax.bar_label(ax.containers[0], fontsize=12, fmt="%.2f")

    ax.set(ylim=(0, 1), ylabel="Proportion", title=title)

    ax.set(title=title)
    ax.set_xticklabels(get_result_labels(df.result), rotation=45)


def plot_daily_side_bias(df, ax):
    """ "
    DONE
    """
    lr_hits = df.groupby(["sides"]).hits.mean().sort_index()
    lr_hits.plot(kind="bar", color=["darkseagreen", "indianred"], ax=ax)
    lr_hits.fillna(
        0, inplace=True
    )  # for early side poke stages, there can be an nan for one side
    bias = lr_hits["l"] - lr_hits["r"]
    ax.set(
        ylim=(0, 1), title=f"Bias: {np.round(bias, 2)}", xlabel="", ylabel="performance"
    )
    # ax.text(0.1, 0.9, f"Bias: {np.round(bias, 2)}")


def plot_daily_side_counts(df, ax):
    "DONE"
    side_count = df.sides.value_counts().sort_index()

    side_count.plot(kind="bar", color=["darkseagreen", "indianred"], ax=ax)
    lr_ratio = side_count["l"] / side_count["r"]
    ax.set(
        title=f"L to R Ratio: {np.round(lr_ratio,2)}",
        xlabel="",
        ylabel="number of trials",
    )


def plot_daily_side_bias_and_counts(df, ax):
    "NOT USED"
    # get sub data (kind of hacky, but good enough for now)
    side_count = df.sides.value_counts(normalize=True)
    lr_hits = df.groupby(["sides"]).hits.mean()
    lr_hits.fillna(
        0, inplace=True
    )  # for early side poke stages, there can be an nan for one side
    side_info_df = pd.merge(side_count, lr_hits, right_index=True, left_index=True)
    # plot
    side_info_df.plot(kind="bar", color=["gray", "darkgreen"], ax=ax)

    # title info
    bias = lr_hits["l"] - lr_hits["r"]
    lr_ratio = side_count["l"] / side_count["r"]
    print(bias, print(lr_ratio))
    ax.set(
        ylim=(0, 1),
        title=f"LR Ratio: {np.round(lr_ratio,2)} Bias: {np.round(bias, 2)}",
        xlabel="",
    )

    ax.legend(frameon=False, borderaxespad=0)

    ##############################
    ####      WATER/MASS      ####
    ##############################


def plot_daily_water(df, ax, title="", legend=False):
    """
    Quick function for plotting water consumed in rig or pub for a day
    and marking the target with a horizontal line
    """
    # TODO add option for plotting a specific date

    animal_id = df.animal_id.iloc[0]
    date = df.date.iloc[0]

    # get the data
    Water_keys = {"rat": animal_id, "date": date}  # specific to Water table
    pub_volume = (ratinfo.Water & Water_keys).fetch("volume")
    if len(pub_volume) == 0:
        pub_volume = 0
    elif len(pub_volume) > 1:
        pub_volume = pub_volume.max()

    rig_volume = df.water_delivered.sum() / 1000  # convert to mL
    volume_target = fetch_daily_water_target(animal_id, date, verbose=False)

    df = pd.DataFrame(
        {"date": [date], "rig_volume": [rig_volume], "pub_volume": [float(pub_volume)]}
    )

    # plot
    df.set_index("date").plot(kind="bar", stacked=True, color=["blue", "cyan"], ax=ax)
    ax.axhline(y=volume_target, xmin=0.2, xmax=0.8, color="black")
    ax.text(x=-0.45, y=volume_target, s=str(volume_target), fontsize=12)
    for cont in ax.containers:
        _ = ax.bar_label(
            cont, fontsize=12, fmt="%.2f", label_type="center", color="white"
        )

    # legend
    if legend:
        order = [1, 0]
        handles, _ = ax.get_legend_handles_labels()
        labels = ["rig", "pub"]
        ax.legend(
            [handles[i] for i in order], [labels[i] for i in order], loc=(0.9, 0.75)
        )
    else:
        ax.get_legend().remove()

    # aesthetics
    ax.set_xticks([])
    # _ = ax.set_xticklabels(ax.get_xticks(), rotation=45)
    _ = ax.set(xlabel="", ylabel="volume (mL)", title=title)
    sns.despine()


def plot_daily_delay_dur(df, ax, title=""):
    was_early_poke = df["valid_early_spoke"].apply(lambda row: 0.1 if row else np.nan)

    sns.lineplot(data=df, x="trial", y="delay_dur", ax=ax)
    ax.scatter(
        x=df.trial, y=was_early_poke, marker="s", color="orangered", label="was early"
    )

    ax.legend(frameon=False, borderaxespad=0)

    ax.set(title=title, ylabel="Delay [s]")


def create_early_poke_type_col(df):
    """
    Function to make a new column where a valid early
    spoke is marked as a 1 and a violation is marked as
    a 2 and 0 otherwise.
    """
    # define the conditions for the new column
    df = df.copy()

    df.loc[:, "violations"].fillna(0, inplace=True)

    conditions = [
        (df["valid_early_spoke"] == 0) & (df["violations"] == 0).astype(bool),
        (df["valid_early_spoke"] == 1).astype(bool),
        (df["violations"] == 1).astype(bool),
    ]

    # define the values for the new column based on the conditions
    values = [0, 1, 2]

    # use numpy.select() to create the new column
    df["early_spoke_type"] = np.select(conditions, values)
    return df


def plot_delay_poke_hist(df, ax, title=""):
    df = create_early_poke_type_col(df)

    sns.histplot(
        data=df,
        x="delay_dur",
        hue="early_spoke_type",
        palette=get_early_poke_colors(df),
        element="step",
        ax=ax,
        legend=True,
    )
    ax.axvline(x=df.exp_del_tau.mean(), color="black", label="Tau")

    ax.set(xlabel="Delay [s]", title=title)
    ax.legend(frameon=False, borderaxespad=0)


def plot_daily_first_spoke_hist(df, ax, title=""):
    first_poke_df = pd.melt(
        df, id_vars=["trial", "first_spoke"], value_vars=["first_lpoke", "first_rpoke"]
    )

    sns.histplot(
        data=first_poke_df,
        x="value",
        hue="first_spoke",
        hue_order=["l", "r"],
        palette=["darkseagreen", "indianred"],
        element="step",
        ax=ax,
        legend=False,
    )

    ax.set(title=title, xlabel="Time to spoke [s]")


def plot_daily_viol_pre_go_durs(df, ax, title=""):
    sns.lineplot(
        data=df, x="trial", y="pre_go_dur", label="pre go", color="teal", ax=ax
    )
    sns.lineplot(
        data=df, x="trial", y="viol_off_dur", label="viol off", color="orange", ax=ax
    )

    was_violation = df["result"].apply(lambda row: True if row == 3.0 else np.nan)
    ax.scatter(x=df.trial, y=was_violation, marker="s", color="orangered")

    ax.set(ylabel="Duration [s]")
    ax.legend(frameon=False, borderaxespad=0)


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
    2: {"label": "error", "color": "maroon"},
    3: {"label": "viol", "color": "orangered"},
    4: {"label": "terr", "color": "lightcoral"},
    1: {"label": "hit", "color": "darkgreen"},
    5: {"label": "crash", "color": "cyan"},
    6: {"label": "noans", "color": "black"},
}

EARLY_POKE_MAP = {
    "none": {"color": "turquoise"},
    "viol": {"color": "orangered"},
    "early": {"color": "gold"},
}


def get_poke_colors(poke_column):
    pokes = poke_column.unique()  # any colum with 'l', 'r', or 'c'
    colors = [POKE_MAP[poke]["color"] for poke in pokes]
    return colors


def get_poke_labels(poke_column):
    pokes = poke_column.sort_values().unique()  # any colum with 'l', 'r', or 'c'
    colors = [POKE_MAP[poke]["label"] for poke in pokes]
    return colors


def get_result_labels(result_column):
    results = result_column.sort_values().unique()
    labels = [RESULT_MAP[res]["label"] for res in results]
    return labels


def get_result_colors(result_column):
    # sorted to match hue_order
    results = result_column.sort_values().unique()
    colors = [RESULT_MAP[res]["color"] for res in results]
    return colors


def get_result_order(result_column):
    return result_column.sort_values().unique()


def get_early_poke_colors(df):
    keys = ["none"]
    if df.violations.sum() > 0:
        keys.append("viol")

    if df.valid_early_spoke.sum() > 0:
        keys.append("early")

    colors = [EARLY_POKE_MAP[key]["color"] for key in keys]

    return colors


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
