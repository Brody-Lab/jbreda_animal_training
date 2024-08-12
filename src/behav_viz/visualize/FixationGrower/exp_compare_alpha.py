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


def plot_stage_in_stage_by_animal_single_experiment(
    df, experiment, ax=None, min_stage=None, max_stage=None, title=""
):

    if ax is None:
        fig, ax = pu.make_fig((8, 4))

    plot_df = df[df["fix_experiment"].str.contains(experiment, case=False)].copy()
    pal = pu.ALPHA_V1_palette if "1" in experiment else pu.ALPHA_V2_palette

    viz.multianimal_plots.plot_ma_days_in_stage_by_animal(
        plot_df,
        ax=ax,
        min_stage=min_stage,
        max_stage=max_stage,
        palette=pal,
        title=title,
    )

    return None


###################### DURATION TO REACH TARGET FIXATION  ######################


def plot_days_to_reach_target_fix_boxplot_compare_experiment(
    df, ax=None, title="", relative_stage=5
):
    """
    Plots a comparison of days to reach the target fixation for different
    experiments using a boxplot and swarmplot.

    Parameters:
        df (DataFrame):
            DataFrame containing the data with columns like "fix_experiment" and "days_to_target".
        ax (matplotlib.axes._subplots.AxesSubplot, optional):
            Axes object to draw the plot onto, otherwise creates a new figure.
        title (str, optional):
            Title for the plot.
        relative_stage (int, optional):
            Which stage to consider for the relative days calculation.

    """

    # Compute target fix df
    target_fix_df = viz.FixationGrower.df_preperation.compute_days_to_target_fix_df(
        df, relative_stage=relative_stage
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # order it so V1 is first on x axis
    target_fix_df["fix_experiment"] = pd.Categorical(
        target_fix_df["fix_experiment"], categories=["V1", "V2"], ordered=True
    )
    # Boxplot
    sns.boxplot(
        data=target_fix_df,
        x="fix_experiment",
        y="days_to_target",
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        dodge=False,
        ax=ax,
        boxprops=dict(alpha=0.5),
    )

    # Swarmplot
    sns.swarmplot(
        data=target_fix_df,
        x="fix_experiment",
        y="days_to_target",
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        ax=ax,
        size=7,
    )

    # Set labels, title, and limits
    ax.set(ylabel="Days to Target Fix", xlabel="", ylim=(3, None), title=title)

    # Remove the extra legend if hue was used in both plots
    if ax.get_legend():
        ax.legend_.remove()

    return None


def plot_days_to_reach_target_fix_histogram_single_experiment(
    df,
    experiment,
    ax=None,
    title="",
    relative_stage=5,
    binwidth=0.9,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    plot_df = df[df["fix_experiment"].str.contains(experiment, case=False)].copy()
    color = pu.ALPHA_V1_color if "1" in experiment else pu.ALPHA_V2_color

    viz.multianimal_plots.days_to_reach_target_fix_histogram(
        plot_df,
        ax=ax,
        relative_stage=relative_stage,
        title=title,
        binwidth=binwidth,
        color=color,
    )

    return None


def plot_days_to_reach_target_fix_histogram_compare_experiment(
    df, ax=None, title="", relative_stage=5, binwidth=0.9
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    viz.multianimal_plots.days_to_reach_target_fix_histogram(
        df,
        ax=ax,
        relative_stage=relative_stage,
        title=title,
        binwidth=binwidth,
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        element="step",
    )
