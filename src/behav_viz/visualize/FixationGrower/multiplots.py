"""
Author: Jess Breda
Date: July 22, 2024
Description: given days, trials table info make summary plots
for an animal for over days. Specifically for Fixation Grower protocol.
"""

import matplotlib.pyplot as plt

from behav_viz.visualize.plot_days_info import *
from behav_viz.visualize.plot_trials_info import *


# Next step - get a multi plot frame work in here as outlined in git and then
# add the new raw/relative fix dur plot to the multiplot with other made plots
# then, done for the day


def over_days_summary(animal_id, animal_days_df, animal_trials_df):
    """
    params
    ------
    animal_id : str
        animal id to plot, e.g. "R610"
    days_df : pd.DataFrame
        days dataframe created by create_days_df_from_dj() with
        day as row index
    trials_df : pd.DataFrame
        trials dataframe created by create_trials_df_from_dj() with
        trial as row index

    """
    layout = """
        AAABBBCCC
        DDDEEEFFF
        GGGHHHIII
        JJJKKKLLL
        MMMNNNOOO
        """
    letters = list(set(layout.replace("\n", "").replace(" ", "")))
    letters.sort()
    bar_plots = list("CEIL")  # manual input axes with bar plots
    bottom_row = letters[-3:]

    fig = plt.figure(constrained_layout=True, figsize=(30, 20))
    ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
    plt.suptitle(f"{animal_id} Daily Summary Plot", fontweight="semibold")

    # determine training stage info
    current_stage = animal_trials_df.stage.iloc[-1]
    current_sma = animal_trials_df.SMA_set.iloc[-1]

    letters = list(set(layout.replace("\n", "").replace(" ", "")))
    bar_plots = list("CEIL")  # manual input axes with bar plots
    bottom_row = letters[-3:]

    ## ROW 1
    plot_trials(animal_days_df, ax_dict["A"], title="Trials", legend=True)
    plot_mass(animal_days_df, ax_dict["B"], title="Mass")
    plot_water_restriction(animal_days_df, ax_dict["C"], title="Water", legend=False)

    ## ROW 2
    plot_performance(animal_days_df, ax_dict["D"], title="Performance")
    plot_performance_bars(animal_trials_df, ax_dict["E"], title="Performance")
    plot_stage(animal_trials_df, ax_dict["F"], title="Stage")

    ## ROW 3
    plot_side_bias(animal_days_df, ax_dict["G"], title="Side Bias")
    plot_antibias_probs(animal_trials_df, ax_dict["H"], title="Antibias")
    plot_sidebias_params(animal_trials_df, ax_dict["I"], title="Side Bias Params")

    ## ROW 4
    plot_time_to_spoke(animal_trials_df, ax_dict["J"], title="Time to Spoke")
    if current_sma == "cpoke":
        plot_cpoke_dur_timings_pregnp(animal_trials_df, ax_dict["K"], title="Cpoke Dur")
        plot_trial_structure(animal_trials_df, ax_dict["L"], title="Trial Structure")

    # ROW 5
    if current_sma == "cpoke":
        plot_give_info_days(
            animal_trials_df,
            ax_dict["M"],
            title="Give Type",
            legend=False,
        )
        plot_n_cpokes_and_multirate(
            animal_trials_df, ax_dict["N"], title="Multi Cpokes"
        )

    plot_rig_tech(animal_days_df, ax_dict["O"], title="Rig Tech")

    pu.adjust_mosaic_axes(ax_dict, letters, bar_plots, bottom_row, animal_days_df)

    return None
