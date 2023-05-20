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
        legend=legend,
    )
    ax.tick_params(axis="x", labelrotation=45)
    _ = ax.set(ylabel="Rate", xlabel="", title=title)
    if legend:
        ax.legend(loc="best", frameon=False, borderaxespad=0)
    ax.set(ylim=(0, 1))


####################
###    STAGE     ###
####################
def plot_stage(df, ax, **kwargs):
    sns.lineplot(
        data=df.groupby("date").mean().stage, drawstyle="steps-post", ax=ax, **kwargs
    )

    _ = plt.xticks(rotation=45)
    _ = plt.yticks(np.arange(1, 11, 1))
    _ = ax.set(ylabel="stage number", title="")
    sns.despine()


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
        side_perf = date_df.groupby("sides").mean().hits.reset_index()

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

    _ = plt.xticks(rotation=45)
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

    ax.tick_params(axis="x", labelrotation=45)
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

    ax.tick_params(axis="x", labelrotation=45)
    ax.set(title="Antibias L/R probs", xlabel="", ylabel="Probability")
    ax.legend(frameon=False, borderaxespad=0)


def plot_multiday_antibias_beta(df, ax):
    """
    TODO add gridlines
    """

    sns.barplot(
        data=df, x="date", y="ab_beta", color="cornflowerblue", errorbar=None, ax=ax
    )
    ax.tick_params(axis="x", labelrotation=45)
    _ = ax.set(ylim=(-0.5, df.ab_beta.max() + 1), ylabel="Beta", xlabel="")
