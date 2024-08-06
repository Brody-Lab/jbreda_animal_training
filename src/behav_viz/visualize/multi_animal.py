import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behav_viz.utils import plot_utils as pu
import behav_viz.visualize as viz

###################### STAGE  ######################


def plot_ma_stage(
    df, ax=None, x_var="date", title="", ylim=None, rotate_x_labels=False, **kwargs
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
        )

    # plot the mean of the animals
    viz.plots.plot_stage(
        df, x_var=x_var, ax=ax, **kwargs, rotate_x_labels=rotate_x_labels, ylim=ylim
    )
