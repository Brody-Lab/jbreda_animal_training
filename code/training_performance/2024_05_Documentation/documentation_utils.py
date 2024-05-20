import sys

sys.path.append("../training_performance")

import platform
from pathlib import Path
from datetime import datetime, timedelta
import datajoint as dj

dj.blob.use_32bit_dims = True  # necessary for pd.blob read

from create_trials_df import create_trials_df_from_dj
from create_days_df import create_days_df_from_dj, lazy_load_days_summary_df

from multiplot_summaries import *

import plot_utils as pu
import dj_utils as dju
import dir_utils as du

from plot_trials_info import *
from plot_days_info import *

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# settings
meeting_data_path = (
    "/Users/jessbreda/Desktop/github/jbreda_animal_training/data/meetings/"
)

COHORT1A = ["R010", "R011", "R012", "R013", "R014", "R015"]
COHORT1B = ["C214", "C215", "C220", "C221", "C222", "C223"]
COHORT2 = [
    "R020",
    "R021",
    "R022",
    "R023",
    "R024",
    "R025",
    "R026",
    "R027",
    "R028",
    "R029",
]
COHORT3 = [
    "R030",
    "R031",
    "R032",
    "R033",
    "R034",
    "R035",
    "R036",
    "R037",
    "R038",
    "R039",
]

COHORT1BDATEMIN = "2023-08-18"  # previously on pbups


def create_curriculum_summary_plots(
    trials_df, animal_id, stage_min=None, date_min=None, date_max=None
):

    query = "animal_id == @animal_id"
    if date_min:
        date_min = pd.Timestamp(date_min).date()
        query += " and date >= @date_min"
    if date_max:
        date_max = pd.Timestamp(date_max).date()
        query += " and date <= @date_max"
    if stage_min:
        query += " and stage >= @stage_min"

    animal_df = trials_df.query(query).copy()

    # Setup the figure
    fig, axarr = plt.subplots(4, 2, figsize=(30, 17), sharex=True)
    fig.suptitle(animal_id, fontsize=36, fontweight="bold")

    # Plotting functions
    plot_stage(
        animal_df,
        ax=axarr[0, 0],
        group="date",
        aesthetics=False,
        ylim=(9, 17),
        title="Stage",
    )
    plot_perf_over_days_from_trials_df(animal_df, ax=axarr[0, 1])
    plot_give_info_days(animal_df, ax=axarr[1, 0], aesthetics=False)
    try:
        plot_give_use_rate_days(animal_df, ax=axarr[1, 1], aesthetics=False)
    except:
        pass
    plot_give_delay_dur_days_line(
        animal_df,
        aesthetics=False,
        ax=axarr[1, 1],
        title="Give Delivery & Delay Metrics",
    )
    plot_performance_by_stim_over_days(
        animal_df,
        without_give=False,
        ax=axarr[2, 0],
        aesthetics=False,
        confidence_intervals=False,
        title="Stim Perf Including Give",
    )
    plot_performance_by_stim_over_days(
        animal_df,
        without_give=False,
        ax=axarr[2, 1],
        aesthetics=False,
        confidence_intervals=False,
        title="Stim Perf No Give",
    )
    plot_performance_by_pro_anti_over_days(
        animal_df,
        without_give=False,
        ax=axarr[3, 0],
        confidence_intervals=False,
        xaxis_label=True,
        title="Pro Anti Including Give",
    )
    plot_performance_by_pro_anti_over_days(
        animal_df,
        without_give=True,
        ax=axarr[3, 1],
        confidence_intervals=False,
        xaxis_label=True,
        title="Pro Anti No Give",
    )

    plt.tight_layout()

    return None


def plot_perf_over_days_from_trials_df(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        data=df, x="date", y="hits", ax=ax, color="green", label="hits", errorbar=None
    )
    sns.lineplot(
        data=df,
        x="date",
        y="violations",
        ax=ax,
        color="orangered",
        label="viol",
        errorbar=None,
    )

    ax2 = ax.twinx()
    sns.lineplot(
        data=df,
        x="date",
        y="trial",
        estimator="max",
        ax=ax2,
        color="k",
        marker=".",
        label="N trials",
    )
    ax.set(
        ylabel="Performance",
        xlabel="Date",
        title="Overall Performance",
        ylim=(-0.1, 1.1),
    )

    # ax.grid(alpha=0.5)

    return None
