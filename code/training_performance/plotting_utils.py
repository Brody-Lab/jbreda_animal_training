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
