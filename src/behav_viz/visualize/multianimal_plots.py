import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
import behav_viz.visualize as viz


###################### FAILED FIXATIONS & VIOLATIONS  ######################


def plot_failed_fix_median_line(ax, df, **kwargs):

    hue = kwargs.get("hue", None)
    palette = kwargs.get("palette", "Set2")
    color = kwargs.get("color", None)
    hue_order = kwargs.get("hue_order", None)

    if hue:

        if hue_order:
            df = df.sort_values(by=hue)
            categories = hue_order
        elif hue:
            categories = sorted(df[hue].unique())

        for idx, category in enumerate(categories):
            median_value = df[df[hue] == category]["failure_rate"].median()
            pal_color = palette[idx]
            ax.axvline(
                median_value,
                linestyle="--",
                color=pal_color,
            )

            ax.text(
                median_value,
                0.1,
                f"{median_value:.2f}",
                rotation=90,
                verticalalignment="bottom",
                horizontalalignment="right",
            )
    else:
        median_value = df["failure_rate"].median()
        ax.axvline(
            median_value,
            linestyle="--",
            label=f"Median: {median_value:.2f}",
            color=color,
        )
        ax.text(
            median_value,
            0.1,
            f"{median_value:.2f}",
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    return None


def plot_failed_fixation_histogram(
    df,
    settling_in_type="by_poke",
    min_stage=5,
    max_stage=None,
    ax=None,
    title="",
    **kwargs,
):

    # Prepare df and filter as needed
    failed_fix_rates_df = (
        viz.FixationGrower.df_preperation.compute_failed_fixation_rate_df(df)
    )

    failed_fix_rates_df = viz.FixationGrower.df_preperation.filter_failed_fix_df(
        failed_fix_rates_df, min_stage, max_stage, settling_in_type
    )

    # Plot
    if ax is None:
        fig, ax = pu.make_fig("m")

    try:
        sns.histplot(
            data=failed_fix_rates_df,
            x="failure_rate",
            binwidth=0.05,
            ax=ax,
            **kwargs,
        )
    except:
        sns.histplot(
            data=failed_fix_rates_df,
            x="failure_rate",
            binwidth=None,
            ax=ax,
            **kwargs,
        )

    # Plot median lines
    plot_failed_fix_median_line(ax, failed_fix_rates_df, **kwargs)

    # Aesthetics
    _ = ax.set(
        title=title,
        xlabel="Failed Fixation Rate",
        ylabel="Session Count",
        xlim=(0, 1),
    )


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
                alpha=0.25,
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
        fig, ax = pu.make_fig("m")

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

    _ = ax.set(ylabel="N Days", xlabel="Stage", title=title, ylim=(0, None))
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
        title="Exp",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    _ = ax.set(ylabel="N Days", xlabel="Stage", title=title, ylim=(0, None))
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


###################### DURATION TO REACH TARGET FIXATION  ######################
def days_to_reach_target_fix_histogram(
    df,
    ax=None,
    relative_stage=5,
    title="",
    binwidth=0.9,
    **kwargs,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    target_fix_df = viz.FixationGrower.df_preperation.compute_days_to_target_fix_df(
        df, relative_stage=relative_stage
    )

    sns.histplot(
        data=target_fix_df,
        x="days_to_target",
        binwidth=binwidth,
        ax=ax,
        **kwargs,
    )

    _ = ax.set(
        xlabel="Days to reach target fixation",
        ylabel="N Animals",
        title=title,
    )

    return None
