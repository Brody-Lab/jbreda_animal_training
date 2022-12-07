## Functions for plotting animal performance from session data fetched & cleaned
## into data frames from DataJoint

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime as dt

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
    sns.lineplot(data=df, x="date", y="hits", errorbar=None, ax=ax, **kwargs)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="fraction correct", title=title, ylim=[0, 1])
    sns.despine()


def plot_viols(df, ax, title=None, **kwargs):
    title = "Violation Plot" if title is None else title
    sns.lineplot(data=df, x="date", y="violations", errorbar=None, ax=ax, **kwargs)
    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="fraction violation", title=title, ylim=[0, 1])
    sns.despine()


def plot_hits_and_viols(df, ax, title=None):
    title = "Hit & Viol Plot" if title is None else title

    sns.lineplot(data=df, x="date", y="hits", color="seagreen", errorbar=None, ax=ax)
    sns.lineplot(
        data=df, x="date", y="violations", color="firebrick", errorbar=None, ax=ax
    )

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


def plot_pair_violations(df, ax, title=None):
    title = "Pair Viol Plot" if title is None else title

    perf_by_sound = df.pivot_table(
        index="date", columns="sound_pair", values="violations", aggfunc="mean"
    )

    colors = {
        "3.0, 3.0": "skyblue",
        "12.0, 12.0": "steelblue",
        "3.0, 12.0": "thistle",
        "12.0, 3.0": "mediumorchid",
    }

    perf_by_sound.plot.line(color=colors, ax=ax, rot=45, style="--")

    _ = ax.set(ylim=[0, 1], ylabel="fraction violations", title=title)
    sns.despine()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0, frameon=False)


def single_day_pair_perf(df, ax):

    latest_df = df[df.date == df.date.max()]

    # this sorting is necessary to keep colors and count labeling correct
    latest_df = latest_df.sort_values(by="sound_pair", key=sound_pairs_sorter)

    palette = create_palette_given_sounds(latest_df)
    sound_pair_counts = latest_df.sound_pair.value_counts(sort=False)
    print(sound_pair_counts)

    sns.barplot(
        data=latest_df,
        x="sound_pair",
        y="hits",
        palette=palette,
        ax=ax,
    )
    # value_counts returns in the reverse order of the sorting done above
    # ax.bar_label(ax.containers[0], sound_pair_counts[::-1], label_type="center")

    ax.set(title=f"{df['animal_id'].iloc[-1]} {df['date'].iloc[-1]} hits")
    sns.despine()
    # return sound_pair_counts


def single_day_pair_viols(df, ax):

    latest_df = df[df.date == df.date.max()]

    # this sorting is necessary to keep colors and count labeling correct
    latest_df = latest_df.sort_values(by="sound_pair", key=sound_pairs_sorter)

    palette = create_palette_given_sounds(latest_df)

    sns.barplot(
        data=latest_df,
        x="sound_pair",
        y="violations",
        palette=palette,
        ax=ax,
        alpha=0.5,
    )
    # value_counts returns in the reverse order of the sorting done above
    # ax.bar_label(ax.containers[0], sound_pair_counts[::-1], label_type="center")

    ax.set(title=f"{df['animal_id'].iloc[-1]} {df['date'].iloc[-1]} violations")
    sns.despine()
    # return sound_pair_counts


def single_date_perf_barplot(df, date, plot_type, ax):
    # TODO add in count labels to the plot for reference

    date_df = df[(df.date == date)]

    # this sorting is necessary to keep colors and count labeling correct
    date_df = date_df.sort_values(by="sound_pair", key=sound_pairs_sorter)
    palette = create_palette_given_sounds(date_df)

    if plot_type == "hits":
        perf_metric = "hits"
    elif plot_type == "viols":
        perf_metric = "violations"
    else:
        assert plot_type == "hits" or plot_type == "viols", "unknown perf metric"

    sns.barplot(
        data=date_df,
        x="sound_pair",
        y=perf_metric,
        palette=palette,
        ax=ax,
        alpha=0.5,
    )

    ax.set(
        title=f"{date_df['animal_id'].iloc[-1]} {date_df['date'].iloc[-1]} {perf_metric}"
    )
    sns.despine()


def sound_pairs_sorter(column):

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


def plot_viol_hist(df, ax, title=None, **kwargs):
    title = "Viol Plot" if title is None else title

    # make df
    pd.options.mode.chained_assignment = None
    ep_df = df.query("early_spoke == True")
    ep_df["viol_time"] = ep_df.apply(
        lambda row: row["spoke_in"] - row["go_time"], axis=1
    )
    ep_df = ep_df[ep_df["viol_time"] > -1.5]  # remove outliers

    sns.histplot(data=ep_df, x="viol_time", binwidth=0.025, color="lightcoral", ax=ax)
    ax.axvline(0, color="black", linewidth=3)

    _ = ax.set(xlabel="Time Pre Go Cue [s]", title=title)
    sns.despine()


def stim_pair_plot(
    ax, sound_pairs, title, match_line, vline, hline, xlim=[0, 15], ylim=[0, 15]
):
    #! NOT WORKING
    stim_range = np.unique(sound_pairs)

    colors = create_palette_given_sounds(sound_pairs)

    for sp, c in zip(sound_pairs, colors):
        ax.scatter(sp[0], sp[1], marker=",", s=300, c=c, alpha=0.75)

    if match_line:
        plt.axline((0, 1), slope=1, color="lightgray", linestyle="--")
        plt.axline((1, 0), slope=1, color="lightgray", linestyle="--")

    if vline == "upper":
        plt.axvline(x=8, ymin=0.55, ymax=1, color="dimgray", linestyle="--")

    elif vline == "lower":
        plt.axvline(x=8, ymax=0.5, color="dimgray", linestyle="--")

    elif vline == "full":
        plt.axvline(x=8, ymax=1, color="dimgray", linestyle="--")

    if hline == "left":
        plt.axhline(y=8, xmax=0.5, color="dimgray", linestyle="--")

    elif hline == "right":
        plt.axhline(y=8, xmin=0.55, xmax=1, color="dimgray", linestyle="--")

    elif hline == "full":
        plt.axhline(y=8, xmax=1, color="dimgray", linestyle="--")

    # aesthetics
    ax.set_title(title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xticks(stim_range)
    ax.set_yticks(stim_range)
    sns.despine()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(stim_range)
    ax.set_yticks(stim_range)
    sns.despine()


def filter_for_date_window(df, latest_date=None, n_days_back=None):
    """
    Function to filter a data frame for a given date window and
    return a date frame in that window. If no parameters are
    specified, will return the input data frame

    inputs
    ------
    df : data frame
        data frame to be filtered with a `date` column of
        of type pd_datetime
    latest_date : str or pd_datetime (optional, default = None)
        latest date to include in the window, defaults to
        the latest date in the data frame
    n_days_back : int (optional, default = None)
        number of days back from `latest_date` to include,
        defaults to all days

    returns
    -------
    df : data frame
        date frame that only includes dates starting at the
        `latest_date` until `n_days_back`

    example usage
    ------------
    `filter_for_date_window(df, latest_date='2022-11-3', n_days_back=8)`
    `filter_for_date_window(df, latest_date=None, n_days_back=7)

    """

    # grab everything prior to specified date.
    # if not specified, use the latest date
    if latest_date:
        latest_date = pd.to_datetime(latest_date)
    else:
        latest_date = df.date.max()
    df = df[(df.date <= latest_date)]

    # starting from latest_date, grab n_days_back
    if n_days_back:
        earliest_date = latest_date - pd.Timedelta(days=n_days_back)
        df = df[(df.date > earliest_date)]

    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def calculate_bias_history(df, latest_date=None, n_days_back=None):
    # select date range
    df = filter_for_date_window(df, latest_date=latest_date, n_days_back=n_days_back)

    # calculate bias
    bias_df = pd.DataFrame(columns=["date", "bias"])

    for date, date_df in df.groupby("date"):

        side_perf = date_df.groupby("sides").mean().hits.reset_index()

        # always make sure left row is first
        side_perf = side_perf.sort_values(by="sides")

        # left hits - right hits, 2 decimal points
        bias = round(side_perf.hits.iloc[0] - side_perf.hits.iloc[1], 2)

        bias_df = pd.concat(
            [bias_df, pd.DataFrame({"date": [date], "bias": [bias]})], ignore_index=True
        )

    return bias_df


def plot_bias_history(df, ax, latest_date=None, n_days_back=None, **kwargs):

    bias_df = calculate_bias_history(
        df, n_days_back=n_days_back, latest_date=latest_date
    )

    # if having issues with time plots, try this
    # bias_df["date"] = bias_df["date"].astype(str)

    sns.lineplot(
        data=bias_df,
        x="date",
        y="bias",
        errorbar=None,
        marker="o",
        markersize=7,
        ax=ax,
        **kwargs,
    )
    ax.axhline(0, color="k", linestyle="--", zorder=1)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="<-- right bias | left bias -->", title="Side Bias", ylim=[-1, 1])
    sns.despine()
    return bias_df


def plot_stim_in_use(df, ax):

    df = df[df.date == df.date.max()]

    stim_pairs_str = df.sound_pair.unique()
    stim_pairs = []

    # iterate over list of strings and reformat into floats
    # for plotting
    for sp in stim_pairs_str:
        sa, sb = sp.split(", ")  # ['sa, sb'] -> 'sa', 'sb'
        stim_pairs.append((float(sa), float(sb)))

    # assigns specific colors to sound pairs to keep theme
    color_palette = create_palette_given_sounds(df)

    for s, c in zip(stim_pairs, color_palette):
        ax.scatter(s[0], s[1], marker=",", s=300, c=c, alpha=0.75)

    # Match/non-match boundary line
    plt.axline((0, 1), slope=1, color="lightgray", linestyle="--")
    plt.axline((1, 0), slope=1, color="lightgray", linestyle="--")

    # plot range & aesthetics
    sp_min, sp_max = np.min(stim_pairs), np.max(stim_pairs)
    stim_range = [sp_min, sp_max]
    x_lim = [sp_min - 3, sp_max + 3]
    y_lim = [sp_min - 3, sp_max + 3]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xticks(stim_range)
    ax.set_yticks(stim_range)
    ax.set(title="Stimulus Pairs", xlabel="Sa [kHz]", ylabel="Sb [kHz]")
    sns.despine()
