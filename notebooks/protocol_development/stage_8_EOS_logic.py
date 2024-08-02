"""
Function related to the notebook determine_stage_8_EOS_logic.ipynb

Written by Jess Breda 2024-08-01
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def plot_metric_over_stage_8(
    df: pd.DataFrame, y_var: str, set_kwargs: Dict, color="maroon"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the failed fixation rate relative to the previous day for each animal.

    Parameters:
    - delta_df (pd.DataFrame): DataFrame containing the data to plot. Must include
    columns 'animal_id' 'stage', and 'days_relative_to_8'. as well as the y_var
    - y_var (str): The name of the column to be used as the y-variable in the plot.
    - set_kwargs (Dict): Dictionary of keyword arguments to set for the plot axes.

    Returns:
    - Tuple[plt.Figure, plt.Axes]: The figure and axes objects of the generated plots.
    """

    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    plot_df = df.query(
        "stage < 9 and days_relative_to_8 > -3 and days_relative_to_8 < 5"
    )

    sns.lineplot(
        data=plot_df,
        x="days_relative_to_8",
        y=y_var,
        marker="o",
        palette=sns.color_palette(["gray"], n_colors=plot_df.animal_id.nunique()),
        hue="animal_id",
        ax=ax[0],
        errorbar=None,
        zorder=100,
    )

    sns.lineplot(
        data=plot_df,
        x="days_relative_to_8",
        y=y_var,
        marker="o",
        color=color,
        alpha=0.5,
        ax=ax[0],
        zorder=100,
    )

    _ = ax[0].set(xlabel="Days in Stage 8", **set_kwargs)

    ax[0].get_legend().remove()
    ax[0].axvline(-0.5, color="black", linestyle="--")

    sns.swarmplot(
        data=plot_df,
        x="days_relative_to_8",
        y=y_var,
        color="gray",
        ax=ax[1],
    )
    sns.boxplot(
        data=plot_df,
        x="days_relative_to_8",
        y=y_var,
        showfliers=False,
        fill=False,
        color=color,
        ax=ax[1],
    )

    _ = ax[1].set(
        xlabel="Days in Stage 8",
        ylabel="",
        title="",
    )

    for a in ax:
        a.axhline(0, color="black", linestyle="--")
        a.axhline(1, color="black", linestyle="--")

    plt.tight_layout()

    return None


def rolling_avg(group, window_size, column="failed_fixation_rate"):
    """
    Helper function for rolling window to be applied to a group.
    This computes a rolling average of 'column' over specified 'window_size',
    resetting at the start of each group.
    """
    # Calculate rolling average and create a new column for it
    group[f"{column}_rolling_avg_{window_size}"] = (
        group[column].rolling(window=window_size, min_periods=1).mean()
    )
    return group


def plot_rolling_averages(df, window_size, days_range):
    """
    Plots rolling averages of 'failed_fixation_rate' over a specified range of days.
    Each subplot represents a day, and each line represents an animal.
    """
    # Filter the DataFrame for the specified days
    filtered_df = df[df["days_relative_to_8"].isin(days_range)]

    # Prepare the figure with a subplot for each day in the range
    fig, ax = plt.subplots(
        1,
        len(days_range),
        figsize=(len(days_range) * 7, 5),
        dpi=200,
        sharey=True,
        sharex=True,
    )

    if len(days_range) == 1:
        ax = [ax]  # Ensure ax is always a list for consistency in single day plots

    # Plot each day's data in a subplot
    for i, day in enumerate(days_range):
        # Prepare data for the current day
        day_data = filtered_df[filtered_df["days_relative_to_8"] == day]

        # Plotting
        sns.lineplot(
            data=day_data,
            x="trial",
            y=f"failed_fixation_rate_rolling_avg_{window_size}",
            marker=".",
            palette=sns.color_palette(["gray"], n_colors=day_data.animal_id.nunique()),
            hue="animal_id",
            ax=ax[i],
            legend=None,  # Remove legend
        )

        sns.lineplot(
            data=day_data,
            x="trial",
            y=f"failed_fixation_rate_rolling_avg_{window_size}",
            marker=".",
            color="maroon",
            alpha=0.5,
            ax=ax[i],
        )

        ax[i].set_title(f"Day {day} Relative to 8")
        ax[i].set_xlabel("Trial")
        ax[i].set_ylabel("Rolling Avg. of Failed Fix. Rate")

    plt.tight_layout()
    plt.show()
