## Functions for plotting animal performance from session data fetched & cleaned
## into data frames from DataJoint

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# from pathlib import Path
# from datetime import date, timedelta


def plot_stage_and_trials(df, ax, title=None):
    title = "Stage & Trials Plot" if title is None else title
    sns.lineplot(data=df.groupby("date").max().trial, color="darkorange", ax=ax)
    _ = plt.xticks(rotation=45)
    ax.set(ylabel="trials per session", title=title)
    # ax.legend(['hits'])

    ax2 = ax.twinx()
    sns.lineplot(
        data=df.groupby("date").max().stage,
        drawstyle="steps-post",
        color="black",
        ax=ax2,
    )
    ax2.set_yticks(np.arange(1, df["stage"].max() + 1, 1))
    ax2.set(ylabel="stage number")

    ax2.legend(["stage"], borderaxespad=0, frameon=False)


def plot_stage(df, ax, title=None, **kwargs):
    title = "Stage Plot" if title is None else title
    sns.lineplot(
        data=df.groupby("date").max().stage, drawstyle="steps-post", ax=ax, **kwargs
    )

    _ = plt.xticks(rotation=45)
    _ = plt.yticks(np.arange(1, 11, 1))
    _ = ax.set(ylabel="stage number", title=title)
    sns.despine()


def plot_trials(df, ax, title=None, **kwargs):
    title = "Trial Plot" if title is None else title
    sns.lineplot(data=df.groupby("date").max().trial, ax=ax, **kwargs)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="trials per session", title=title)
    sns.despine()


def plot_hits(df, ax, title=None, **kwargs):
    title = "Hit Plot" if title is None else title
    sns.lineplot(data=df, x="date", y="hits", ci=None, ax=ax, **kwargs)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="fraction correct", title=title, ylim=[0, 1])
    sns.despine()


def plot_viols(df, ax, title=None, **kwargs):
    title = "Violation Plot" if title is None else title
    sns.lineplot(data=df, x="date", y="violations", ci=None, ax=ax, **kwargs)
    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="fraction violation", title=title, ylim=[0, 1])
    sns.despine()


def plot_hits_and_viols(df, ax, title=None):
    title = "Hit & Viol Plot" if title is None else title

    sns.lineplot(data=df, x="date", y="hits", color="seagreen", ci=None, ax=ax)
    sns.lineplot(data=df, x="date", y="violations", color="firebrick", ci=None, ax=ax)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="fraction correct | viol", title=title, ylim=[0, 1])
    ax.legend(
        ["hits", "viols"],
        bbox_to_anchor=(1, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )
    sns.despine()


def plot_pair_performance(df, ax, title=None):
    title = "Pair Perf Plot" if title is None else title

    perf_by_sound = df.pivot_table(
        index="date", columns="sound_pair", values="hits", aggfunc="mean"
    )

    colors = {
        "3.0, 3.0": "skyblue",
        "12.0, 12.0": "steelblue",
        "3.0, 12.0": "thistle",
        "12.0, 3.0": "mediumorchid",
    }

    perf_by_sound.plot.line(color=colors, ax=ax, rot=45, style=".-")

    _ = ax.set(ylim=[0, 1], ylabel="fraction correct", title=title)
    sns.despine()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0, frameon=False)


def single_day_pair_perf(df, ax):

    latest_df = df[df.date == df.date.max()]

    # this sorting is necessary to keep colors and count labeling correct
    latest_df = latest_df.sort_values(by="sound_pair", key=_sound_pairs_sorter)

    palette = create_palette_given_sounds(latest_df)
    sound_pair_counts = latest_df.sound_pair.value_counts(sort=False)

    sns.barplot(
        data=latest_df,
        x="sound_pair",
        y="hits",
        palette=palette,
        ax=ax,
    )
    # value_counts returns in the reverse order of the sorting done above
    ax.bar_label(ax.containers[0], sound_pair_counts[::-1], label_type="center")

    ax.set(title=f"{df['animal_id'].iloc[-1]} {df['date'].iloc[-1]}")
    sns.despine()


def _sound_pairs_sorter(column):

    "Function to order df columns by match and then nonmatch sound pairs"

    sp_order = ["3.0, 3.0", "12.0, 12.0", "3.0, 12.0", "12.0, 3.0"]
    correspondence = {sp: order for order, sp in enumerate(sp_order)}
    return column.map(correspondence)


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
