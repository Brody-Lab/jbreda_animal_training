import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
import behav_viz.visualize as viz
from behav_viz.visualize.df_preperation import compute_days_relative_to_stage


def plot_stage_relative_to_stage(
    df,
    stage,
    condition,
    ylim=None,
    ax=None,
    title="",
    **kwargs,
):

    if ax is None:
        fig, ax = pu.make_fig()

    # make the df

    df = df[df["curriculum"].str.contains(condition, case=False)].copy()
    plot_df = compute_days_relative_to_stage(df, stage).reset_index()
    x_var = f"days_relative_to_stage_{stage}"

    # plot
    if ax is None:
        fig, ax = pu.make_fig()

    viz.multi_animal.plot_ma_stage(
        plot_df.query("stage >= @stage"),
        ax=ax,
        x_var=x_var,
        ylim=ylim,
        **kwargs,
    )

    _ = ax.set(
        xlabel=f"Days relative to stage {stage}",
        title=f"Days in Stage for {condition} animals (N = {len(plot_df.animal_id.unique())})",
    )

    return None
