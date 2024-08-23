import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
import behav_viz.visualize as viz
from behav_viz.visualize.df_preperation import compute_days_relative_to_stage


###################### FAILED FIXATION & VIOLATIONS  ######################

#### OVER DAYS ####


def plot_failed_fixation_rate_single_experiment(
    df,
    experiment,
    ax=None,
    title="",
    min_stage=5,
    max_stage=None,
    settling_in_type="by_poke",
    relative_to_stage=5,
):

    if ax is None:
        fig, ax = pu.make_fig()

    plot_df = df[df["fix_experiment"].str.contains(experiment, case=False)].copy()
    color = pu.ALPHA_V1_color if "1" in experiment else pu.ALPHA_V2_color

    viz.multianimal_plots.plot_ma_failed_fixation_rate(
        plot_df,
        ax=ax,
        min_stage=min_stage,
        max_stage=max_stage,
        settling_in_type=settling_in_type,
        relative_to_stage=relative_to_stage,
        title=title,
        color=color,
        style="type",
    )

    return None


def plot_failed_fixation_rate_turn_penalty_on(
    df,
    ax=None,
    title="",
    min_stage=7,
    max_stage=8,
    day_range=(-3, 3),
    settling_in_type="by_poke",
    style="type",
):
    """
    Note this plot is only relevant for the V2 Condition

    """
    if ax is None:
        fig, ax = pu.make_fig()

    plot_df = df[df["fix_experiment"].str.contains("V2", case=False)].copy()
    color = pu.ALPHA_V2_color

    # hacky way to pre-filter the data so x lims are nice
    plot_df = compute_days_relative_to_stage(plot_df, stage=8).reset_index()
    plot_df = plot_df.query(
        f"days_relative_to_stage_8 >= {day_range[0]} & days_relative_to_stage_8 <= {day_range[1]}"
    ).copy()

    viz.multianimal_plots.plot_ma_failed_fixation_rate(
        plot_df,
        ax=ax,
        min_stage=min_stage,
        max_stage=max_stage,
        settling_in_type=settling_in_type,
        title=title,
        color=color,
        relative_to_stage=8,
        style=style,
    )

    ax.grid()
    ax.axvline(-0.5, color="black", linestyle="--")

    return None


def plot_failed_fixation_rate_compare_experiment(
    df,
    ax=None,
    title="",
    min_stage=5,
    max_stage=7,
    settling_in_type="by_poke",
    relative_to_stage=5,
    plot_individuals=True,
):

    if ax is None:
        fig, ax = pu.make_fig()

    # order it so V1 is first on x axis
    df["fix_experiment"] = pd.Categorical(
        df["fix_experiment"], categories=["V1", "V2"], ordered=True
    )

    if plot_individuals:

        viz.multianimal_plots.plot_ma_failed_fixation_rate_by_condition(
            df,
            condition="fix_experiment",
            ax=ax,
            min_stage=min_stage,
            max_stage=max_stage,
            settling_in_type=settling_in_type,
            relative_to_stage=relative_to_stage,
            title=title,
            hue_order=["V1", "V2"],
            palette=pu.ALPHA_PALLETTE,
        )

    else:
        viz.FixationGrower.plots.plot_failed_fixation_rate(
            df,
            ax=ax,
            min_stage=min_stage,
            max_stage=max_stage,
            settling_in_type=settling_in_type,
            relative_to_stage=relative_to_stage,
            hue="fix_experiment",
            hue_order=["V1", "V2"],
            palette=pu.ALPHA_PALLETTE,
            title=title,
        )

    ax.set(ylim=(0, 0.5))

    return None


#### SUMMARIES  ####


def plot_failed_fixation_histogram_single_experiment(
    df,
    experiment,
    ax=None,
    title="",
    min_stage=5,
    max_stage=None,
    settling_in_type="by_poke",
):

    if ax is None:
        fig, ax = pu.make_fig("m")

    plot_df = df[df["fix_experiment"].str.contains(experiment, case=False)].copy()
    color = pu.ALPHA_V1_color if "1" in experiment else pu.ALPHA_V2_color

    viz.multianimal_plots.plot_failed_fixation_histogram(
        plot_df,
        ax=ax,
        min_stage=min_stage,
        max_stage=max_stage,
        settling_in_type=settling_in_type,
        title=title,
        color=color,
    )

    return None


def plot_failed_fixation_histogram_compare_experiment(
    df,
    ax=None,
    title="",
    min_stage=5,
    max_stage=None,
    settling_in_type="by_poke",
):

    if ax is None:
        fig, ax = pu.make_fig("m")

    # order it so V1 is first on x axis
    df["fix_experiment"] = pd.Categorical(
        df["fix_experiment"], categories=["V1", "V2"], ordered=True
    )

    viz.multianimal_plots.plot_failed_fixation_histogram(
        df,
        ax=ax,
        min_stage=min_stage,
        max_stage=max_stage,
        settling_in_type=settling_in_type,
        title=title,
        hue="fix_experiment",
        hue_order=["V1", "V2"],
        palette=pu.ALPHA_PALLETTE,
        element="step",
    )

    return None


###################### STAGE PROGRESS OVER DATE ######################


def plot_stage_compare_experiments(
    df,
    ax=None,
    title="",
    ylim=None,
    rotate_x_labels=False,
    relative_to_stage=None,
    plot_individuals=True,
):

    if ax is None:
        fig, ax = pu.make_fig()

    if plot_individuals:

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
    else:
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
        errorbar=None,
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
        fig, ax = pu.make_fig("m")

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
        fig, ax = pu.make_fig("m")

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
        fig, ax = pu.make_fig("m")

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
