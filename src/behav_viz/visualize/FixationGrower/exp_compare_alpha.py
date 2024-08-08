import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
import behav_viz.visualize as viz
from behav_viz.visualize.df_preperation import compute_days_relative_to_stage


###################### STAGE PROGRESS OVER DATE ######################


def plot_ma_stage_compare_experiments(
    df,
    ax=None,
    title="",
    ylim=None,
    rotate_x_labels=False,
    relative_to_stage=None,
):

    if ax is None:
        fig, ax = pu.make_fig()

    viz.multianimal_plots.plot_ma_stage_by_condition(
        df,
        condition="fix_experiment",
        ax=ax,
        title=title,
        palette=pu.ALPHA_PALLETTE,
        hue_order=["V1", "V2"],
        ylim=ylim,
        rotate_x_labels=rotate_x_labels,
        relative_to_stage=relative_to_stage,
    )

    return None


def plot_ma_stage_single_experiment(
    df,
    experiment,
    ax=None,
    title="",
    ylim=None,
    rotate_x_labels=False,
    relative_to_stage=None,
):

    if ax is None:
        fig, ax = pu.make_fig()

    plot_df = df[df["fix_experiment"].str.contains(experiment, case=False)].copy()
    color = pu.ALPHA_V1_color if "1" in experiment else pu.ALPHA_V2_color

    viz.multianimal_plots.plot_ma_stage(
        plot_df,
        ax=ax,
        x_var="date",
        ylim=ylim,
        title=title,
        rotate_x_labels=rotate_x_labels,
        color=color,
        relative_to_stage=relative_to_stage,
    )

    _ = ax.set(title=title)

    return None


###################### STAGE DURATION ######################
def compare_plot_days_in_stage(
    df,
    ax=None,
    min_stage=None,
    max_stage=None,
):
    """TODO: first make a general one then specific"""

    if ax is None:
        fig, ax = pu.make_fig((6, 4))

    days_in_stage_df = viz.df_preperation.make_days_in_stage_df(
        df, min_stage, max_stage, hue_var="fix_experiment"
    )

    sns.boxplot(
        data=days_in_stage_df,
        x="stage",
        y="n_days",
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=[pu.ALPHA_V1_color, pu.ALPHA_V2_color],
        ax=ax,
        showfliers=False,
        dodge=True,
    )
    sns.swarmplot(
        data=days_in_stage_df,
        x="stage",
        y="n_days",
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=[pu.ALPHA_V1_color, pu.ALPHA_V2_color],
        alpha=0.5,
        dodge=True,
        ax=ax,
    )

    _ = ax.set(ylabel="N Days", xlabel="Stage")
    sns.despine()

    # Optionally adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:2], labels[0:2], title="fix_experiment", frameon=False)

    return None
