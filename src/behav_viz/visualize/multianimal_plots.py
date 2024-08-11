import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
import behav_viz.visualize as viz

###################### STAGE  ######################


def plot_ma_stage(
    df,
    ax=None,
    x_var="date",
    title="",
    ylim=None,
    rotate_x_labels=False,
    relative_to_stage=None,
    **kwargs,
):

    if ax is None:
        fig, ax = pu.make_fig()

    # plot each animal as a gray line
    for _, sub_df in df.groupby("animal_id"):
        viz.plots.plot_stage(
            sub_df,
            x_var=x_var,
            ax=ax,
            alpha=0.5,
            color="gray",
            relative_to_stage=relative_to_stage,
        )

    # plot the mean of the animals
    viz.plots.plot_stage(
        df,
        x_var=x_var,
        ax=ax,
        rotate_x_labels=rotate_x_labels,
        ylim=ylim,
        relative_to_stage=relative_to_stage,
        **kwargs,
    )


def plot_ma_stage_by_condition(
    df,
    condition,
    ax=None,
    x_var="date",
    title="",
    palette="husl",
    ylim=None,
    rotate_x_labels=False,
    relative_to_stage=None,
    **kwargs,
):

    if ax is None:
        fig, ax = pu.make_fig()

    # hacky way of plot multi animals with respective colors
    pal = sns.color_palette(palette, len(df[condition].unique()))
    for ii, (cond, sub_df) in enumerate(df.groupby([condition])):

        color = pal[ii]

        for _, sub_sub_df in sub_df.groupby("animal_id"):
            viz.plots.plot_stage(
                sub_sub_df,
                x_var=x_var,
                ax=ax,
                alpha=0.5,
                color=color,
                relative_to_stage=relative_to_stage,
                **kwargs,
            )

    # plot the mean of the animals
    viz.plots.plot_stage(
        df,
        hue=condition,
        x_var=x_var,
        ax=ax,
        rotate_x_labels=rotate_x_labels,
        ylim=ylim,
        palette=pal,
        relative_to_stage=relative_to_stage,
        **kwargs,
    )

    _ = ax.set(
        ylabel="Stage",
        title=title,
    )

    return None


def plot_ma_days_in_stage(
    df,
    ax=None,
    min_stage=None,
    max_stage=None,
    plot_individuals=True,
    title="",
    **kwargs,
):

    if ax is None:
        fig, ax = pu.make_fig((6, 4))

    days_in_stage_df = viz.df_preperation.make_days_in_stage_df(
        df, min_stage, max_stage
    )

    sns.boxplot(
        data=days_in_stage_df, x="stage", y="n_days", **kwargs, ax=ax, showfliers=False
    )
    if plot_individuals:
        sns.swarmplot(
            data=days_in_stage_df, x="stage", y="n_days", label="", color="gray", ax=ax
        )

    _ = ax.set(ylabel="N Days", xlabel="Stage", title=title)
    sns.despine()

    return None


def plot_ma_days_in_stage_by_condition(
    df,
    condition,
    ax=None,
    min_stage=None,
    max_stage=None,
    plot_individuals=True,
    title="",
    **kwargs,
):

    if ax is None:
        fig, ax = pu.make_fig((6, 4))

    days_in_stage_df = viz.df_preperation.make_days_in_stage_df(
        df,
        min_stage,
        max_stage,
    )

    sns.boxplot(
        data=days_in_stage_df,
        x="stage",
        y="n_days",
        hue=condition,
        **kwargs,
        ax=ax,
        showfliers=False,
        dodge=True,
    )
    if plot_individuals:
        sns.swarmplot(
            data=days_in_stage_df,
            x="stage",
            y="n_days",
            hue=condition,
            ax=ax,
            dodge=True,
            alpha=0.5,
            **kwargs,
        )

    # Only plot legend for boxplot
    n_conditons = len(days_in_stage_df[condition].unique())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[0:n_conditons],
        labels[0:n_conditons],
        title="fix_experiment",
        frameon=False,
    )

    _ = ax.set(ylabel="N Days", xlabel="Stage", title=title)
    sns.despine()

    return None


def plot_ma_days_in_stage_by_animal(
    df,
    ax=None,
    min_stage=None,
    max_stage=None,
    title="",
    **kwargs,
):

    if ax is None:
        fig, ax = pu.make_fig((8, 4))

    days_in_stage_df = viz.df_preperation.make_days_in_stage_df(
        df, min_stage, max_stage
    )

    sns.barplot(
        data=days_in_stage_df,
        x="animal_id",
        y="n_days",
        hue="stage",
        **kwargs,
        ax=ax,
    )

    _ = ax.set(ylabel="N Days", xlabel="", title=title)
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Stage")
    sns.despine()

    return None
