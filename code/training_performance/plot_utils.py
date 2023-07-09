"""
Author: Jess Breda
Date: May 31, 2023
Description: Plotting utilities for training performance
"""

import matplotlib.pyplot as plt


def make_fig(figsize=(10, 3)):
    "Quick fx for subplot w/ 10 x 3 size default"

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


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


### Poke/Side column utilities ###

POKE_MAP = {
    "l": {"label": "left", "color": "darkseagreen"},
    "c": {"label": "center", "color": "khaki"},
    "r": {"label": "right", "color": "indianred"},
    "n": {"label": "", "color": "white"},
    "order": ["l", "c", "r", "n"],
}


def get_poke_labels(poke_column):
    pokes = get_poke_order(poke_column)  # can be any colum with 'l', 'r', or 'c'
    colors = [POKE_MAP[poke]["label"] for poke in pokes]
    return colors


def get_poke_colors(poke_column):
    pokes = get_poke_order(poke_column)  # can any colum with 'l', 'r', or 'c'
    colors = [POKE_MAP[poke]["color"] for poke in pokes]
    return colors


def get_poke_order(poke_column):
    return sorted(
        poke_column.unique(),
        key=lambda x: POKE_MAP["order"].index(x)
        if x in POKE_MAP["order"]
        else float("inf"),
    )
