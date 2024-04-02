"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting utilities for training performance
"""

import matplotlib.pyplot as plt
import seaborn as sns


def make_fig(dims=None):
    "Quick fx for subplot w/ 10 x 3 size default"

    if dims == "s":
        dims = (4, 4)
    elif dims == "w" or dims is None:
        dims = (12, 4)

    fig, ax = plt.subplots(figsize=dims)
    return fig, ax


### Multiplot utilities ###


def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in a multiplot.
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


### Axis utilities ###


def set_date_x_ticks(ax, xaxis_label):
    "Quick fx for rotating xticks on date axis using ax object (not plt.)"

    if xaxis_label:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=45)
    else:  # turn off the labels
        ax.set_xticklabels([])


def set_legend(ax, legend):
    "Quick fx for setting legend on/off using ax object (not plt.)"
    if legend:
        ax.legend(frameon=False, borderaxespad=0)
    else:
        ax.get_legend().remove()


### Result column utilities ###

RESULT_MAP = {
    1: {"label": "hit", "color": "darkgreen"},
    2: {"label": "error", "color": "maroon"},
    3: {"label": "viol", "color": "orangered"},
    4: {"label": "terr", "color": "lightcoral"},
    5: {"label": "crash", "color": "cyan"},
    6: {"label": "noans", "color": "black"},
}


def get_result_labels(result_column):
    results = get_result_order(result_column)
    labels = [RESULT_MAP[res]["label"] for res in results]
    return labels


def get_result_colors(result_column):
    # sorted to match hue_order
    results = get_result_order(result_column)
    colors = [RESULT_MAP[res]["color"] for res in results]
    return colors


def get_result_order(result_column):
    return result_column.sort_values().unique()


### SIDE/Side column utilities ###

SIDE_MAP = {
    "l": {"label": "left", "color": "darkseagreen"},
    "c": {"label": "center", "color": "khaki"},
    "r": {"label": "right", "color": "indianred"},
    "n": {"label": "", "color": "white"},
    "order": ["l", "c", "r", "n"],
    # "n_lpokes": {"label": "left", "color": "darkseagreen"},
    # "n_cpokes": {"label": "center", "color": "khaki"},
    # "n_rpokes": {"label": "right", "color": "indianred"},
}


def get_side_labels(side_column):
    """
    get the correct plot label (left, right, center) location
    for each side

    params:
    -------
    side_column: pandas.Series
        a column of the dataframe that contains the side
        of the poke ('l', 'r', or 'c') or variable relating
        to side ('n_lpokes', 'n_rpokes', 'n_cpokes'
    """
    sides = get_side_order(side_column)  # can be any colum with 'l', 'r', or 'c'
    # sides = side_column.unique()
    labels = [SIDE_MAP[side]["label"] for side in sides]
    return labels


def get_side_colors(side_column):
    """
    get the correct plot color (red, yellow, green) for each side

    params:
    -------
    side_column: pandas.Series
        a column of the dataframe that contains the side
        of the poke ('l', 'r', or 'c') or variable relating
        to side ('n_lpokes', 'n_rpokes', 'n_cpokes'
    """
    sides = get_side_order(side_column)  # can any colum with 'l', 'r', or 'c'
    # sides = side_column.unique()
    colors = [SIDE_MAP[side]["color"] for side in sides]
    return colors


def get_side_order(side_column):
    """
    order the side in L-R-C order for plotting
    params:
    -------
    side_column: pandas.Series
        a column of the dataframe that contains the side
        of the poke ('l', 'r', or 'c') or variable relating
        to side ('n_lpokes', 'n_rpokes', 'n_cpokes'
    """
    return sorted(
        side_column.unique(),
        key=lambda x: (
            SIDE_MAP["order"].index(x) if x in SIDE_MAP["order"] else float("inf")
        ),
    )


### TRIAL DURATION utilities ###

trial_period_palette = sns.color_palette("Set3", 6).as_hex()


TRIAL_PERIOD_MAP = {
    "settling_in": trial_period_palette[0],
    "adj_pre": trial_period_palette[1],
    "s_a": trial_period_palette[2],
    "delay": trial_period_palette[3],
    "s_b": trial_period_palette[4],
    "post": trial_period_palette[5],
    "order": ["settling_in", "adj_pre", "s_a", "delay", "s_b", "post", "go"],
    "go": "white",
}


def get_period_order(trial_period_column):
    """
    order the trial period in settling_in_dur,
    adj_pre_dur, sa, delay_dur, sb, post_dur order for plotting

    params:
    -------
    trial_period_column: pandas.Series
        a column of the dataframe that contains the trial period
        of the poke ('settling_in_dur', 'adj_pre_dur', 'sa',
        'delay_dur', 'sb', 'post_dur')
    """
    return sorted(
        trial_period_column.unique(),
        key=lambda x: (
            TRIAL_PERIOD_MAP["order"].index(x)
            if x in TRIAL_PERIOD_MAP["order"]
            else float("inf")
        ),
    )


def get_period_colors(trial_period_column):
    """
    get the correct plot color for the trial period

    params:
    -------
    trial_period_column: pandas.Series
        a column of the dataframe that contains the trial period
        of the poke ('settling_in_dur', 'adj_pre_dur', 'sa',
        'delay_dur', 'sb', 'post_dur')
    """
    periods = get_period_order(trial_period_column)
    print(periods)
    colors = [TRIAL_PERIOD_MAP[period] for period in periods]
    return colors


### GIVE utilities ###


GIVE_MAP = {
    "none": "green",
    "n": "green",
    "light": "#fcba03",
    "l": "#fcba03",
    "water_and_light": "cornflowerblue",
    "w + l": "cornflowerblue",
    "water": "cyan",
    "w": "cyan",
    "order": ["none", "n", "light", "l", "water_and_light", "w + l", "water", "w"],
}


def get_give_order(trial_give_column):
    """
    order the give types for plotting

    params:
    -------
    trial_give_columb: pandas.Series
        a column of the dataframe that contains the give
        type for a trial
    """
    return sorted(
        trial_give_column.unique(),
        key=lambda x: (
            GIVE_MAP["order"].index(x) if x in GIVE_MAP["order"] else float("inf")
        ),
    )


def get_give_colors(trial_give_column):
    """
    color the give types for plotting

    params:
    -------
    trial_give_column: pandas.Series
        a column of the dataframe that contains the give
        type for a trial
    """
    gives = get_give_order(trial_give_column)
    colors = [GIVE_MAP[give] for give in gives]
    return colors


### STIMULUS utilities ###


def create_palette_given_sounds(df):
    """
    Function to allow for assignment of specific colors to a sound pair
    that is consistent across sessions where number of unique pairs varies
    """
    palette = []
    sound_pairs = df.sound_pair.unique()

    sound_pair_colormap = {
        "3.0, 3.0": "skyblue",
        "12.0, 12.0": "steelblue",
        "3.0, 12.0": "thistle",
        "12.0, 3.0": "mediumorchid",
    }

    for sp in sound_pairs:
        palette.append(sound_pair_colormap[sp])
    return palette


## GIVE & BLOCK SWITCH utilities ###


def get_block_switch_and_give_order():
    return ["sampled", "static", "none", "light", "water", "light_and_water"]


def get_block_switch_and_give_colors():
    return ["salmon", "brown", "green", "gold", "cyan", "cornflowerblue"]
