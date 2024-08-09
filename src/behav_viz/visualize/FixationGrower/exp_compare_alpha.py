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


def plot_stage_compare_experiment(
    df,
    ax=None,
    title="",
    ylim=None,
    rotate_x_labels=False,
    relative_to_stage=None,
):

    if ax is None:
        fig, ax = pu.make_fig()

    viz.plots.plot_stage(
        df,
        ax=ax,
        hue="fix_experiment",
        palette=pu.ALPHA_PALLETTE,
        rotate_x_labels=rotate_x_labels,
        ylim=ylim,
        relative_to_stage=relative_to_stage,
    )

    return None


###################### STAGE DURATION ######################
def plot_days_in_stage_single_experiment(
    df,
    experiment,
    ax=None,
    min_stage=None,
    max_stage=None,
    title="",
):

    if ax is None:
        fig, ax = pu.make_fig((6, 4))

    plot_df = df[df["fix_experiment"].str.contains(experiment, case=False)].copy()
    color = pu.ALPHA_V1_color if "1" in experiment else pu.ALPHA_V2_color

    viz.multianimal_plots.plot_ma_days_in_stage(
        plot_df,
        ax=ax,
        min_stage=min_stage,
        max_stage=max_stage,
        plot_individuals=True,
        color=color,
        title=title,
    )

    return None


def plot_days_in_stage_compare_experiment(
    df, ax=None, min_stage=None, max_stage=None, title=""
):
    """TODO: first make a general one then specific"""

    if ax is None:
        fig, ax = pu.make_fig((6, 4))

    viz.multianimal_plots.plot_ma_days_in_stage_by_condition(
        df,
        condition="fix_experiment",
        ax=ax,
        min_stage=min_stage,
        max_stage=max_stage,
        plot_individuals=True,
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        title=title,
    )

    return None
