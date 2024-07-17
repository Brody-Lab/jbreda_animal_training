"""
Author: Jess Breda
Date: July 17, 2024
Description: given days, trials table info make summary plots
for an animal for over days. Specifically for DMS2 protocol.
"""

import matplotlib.pyplot as plt

from behav_viz.visualize.plot_days_info import *
from behav_viz.visualize.plot_trials_info import *


def over_days_summary_pre_pro_anti(animal_id, animal_days_df, animal_trials_df):
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
    fig = plt.figure(constrained_layout=True, figsize=(30, 20))
    ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
    plt.suptitle(f"{animal_id} Daily Summary Plot", fontweight="semibold")

    # determine training stage info
    current_stage = animal_trials_df.stage.iloc[-1]
    current_sma = animal_trials_df.SMA_set.iloc[-1]

    ## ROW 1
    plot_trials(
        animal_days_df, ax_dict["A"], title="Trials", legend=True, xaxis_label=False
    )
    plot_mass(animal_days_df, ax_dict["B"], title="Mass", xaxis_label=False)
    plot_water_restriction(
        animal_days_df, ax_dict["C"], title="Water", legend=False, xaxis_label=False
    )

    ## ROW 2
    plot_performance(
        animal_days_df, ax_dict["D"], title="Performance", xaxis_label=False
    )
    plot_performance_bars(
        animal_trials_df, ax_dict["E"], title="Performance", xaxis_label=False
    )
    plot_stage(
        animal_trials_df,
        ax_dict["F"],
        title="Stage",
        xaxis_label=False,
    )

    ## ROW 3
    plot_side_bias(animal_days_df, ax_dict["G"], title="Side Bias", xaxis_label=False)
    plot_antibias_probs(
        animal_trials_df, ax_dict["H"], title="Antibias", xaxis_label=False
    )
    plot_sidebias_params(
        animal_trials_df, ax_dict["I"], title="Side Bias Params", xaxis_label=False
    )

    ## ROW 4
    if current_stage == 9:
        plot_sounds_info(
            animal_trials_df, ax_dict["J"], title="Sounds", xaxis_label=False
        )
    elif current_stage == 10 or current_stage == 11:
        plot_performance_by_stim_over_days(
            animal_trials_df,
            without_give=True,
            ax=ax_dict["J"],
            title="Non-Give Perf",
            xaxis_label=False,
        )
    else:
        plot_time_to_spoke(
            animal_trials_df, ax_dict["J"], title="Time to Spoke", xaxis_label=False
        )
    # j is perf either by give or stim depd
    # k stays, # l becomes timing
    # k cpoke dur, # l timing
    # m becomes give metric or pro/anti perf
    # n is multi cpokes or pro/anti
    # o is rig/tech
    if current_sma == "cpoke":
        plot_cpoke_dur_timings_pregnp(
            animal_trials_df, ax_dict["K"], title="Cpoke Dur", xaxis_label=False
        )
        plot_trial_structure(
            animal_trials_df, ax_dict["L"], title="Trial Structure", xaxis_label=False
        )

    ## ROW 5
    if current_sma == "cpoke":
        if current_stage == 10 or current_stage == 11:
            plot_performance_by_give(
                animal_trials_df, ax_dict["M"], title="Give Metrics", xaxis_label=True
            )

            plot_n_cpokes_and_multirate(
                animal_trials_df, ax_dict["N"], title="Multi Cpokes", xaxis_label=True
            )
        else:
            plot_give_info_days(
                animal_trials_df,
                ax_dict["M"],
                title="Give Type",
                xaxis_label=True,
                legend=False,
            )
            plot_n_cpokes_and_multirate(
                animal_trials_df, ax_dict["N"], title="Multi Cpokes", xaxis_label=True
            )

    plot_rig_tech(animal_days_df, ax_dict["O"], title="Rig Tech", xaxis_label=True)

    return None


def over_days_summary_pro_anti(animal_id, animal_days_df, animal_trials_df):
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
        PPPQQQRRR
        SSSTTTUUU
    """
    fig = plt.figure(constrained_layout=True, figsize=(30, 25))
    ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
    plt.suptitle(f"{animal_id} Daily Summary Plot", fontweight="semibold")

    # determine training stage info
    current_stage = animal_trials_df.stage.iloc[-1]
    current_sma = animal_trials_df.SMA_set.iloc[-1]
    latest_date = animal_trials_df.date.iloc[-1]

    if animal_trials_df.query("date == @latest_date").give_delay_dur.nunique() > 2:
        plot_give_del_info = True
    else:
        plot_give_del_info = False

    ## ROW 1
    plot_trials(
        animal_days_df, ax_dict["A"], title="Trials", legend=True, xaxis_label=False
    )
    plot_mass(animal_days_df, ax_dict["B"], title="Mass", xaxis_label=False)
    plot_water_restriction(
        animal_days_df, ax_dict["C"], title="Water", legend=False, xaxis_label=False
    )

    ## ROW 2
    plot_performance(
        animal_days_df, ax_dict["D"], title="Performance", xaxis_label=False
    )
    plot_performance_bars(
        animal_trials_df, ax_dict["E"], title="Performance", xaxis_label=False
    )
    plot_stage(
        animal_trials_df,
        ax_dict["F"],
        title="Stage",
        xaxis_label=False,
    )

    ## ROW 3
    plot_side_bias(animal_days_df, ax_dict["G"], title="Side Bias", xaxis_label=False)
    plot_antibias_probs(
        animal_trials_df, ax_dict["H"], title="Antibias", xaxis_label=False
    )
    plot_sidebias_params(
        animal_trials_df, ax_dict["I"], title="Side Bias Params", xaxis_label=False
    )

    ## ROW 4- JKL
    plot_performance_by_stim_over_days(
        animal_trials_df,
        without_give=False,
        ax=ax_dict["J"],
        title="Stim Perf",
        xaxis_label=False,
        confidence_intervals=False,
    )
    try:
        plot_performance_by_stim_over_days(
            animal_trials_df,
            without_give=True,
            ax=ax_dict["K"],
            title="Non-Give Stim Perf",
            xaxis_label=False,
        )
    except:
        plot_performance_by_stim_over_days(
            animal_trials_df,
            without_give=True,
            ax=ax_dict["K"],
            title="Non-Give Stim Perf",
            xaxis_label=False,
            confidence_intervals=False,
        )

    plot_trial_structure(
        animal_trials_df, ax_dict["L"], title="Trial Structure", xaxis_label=False
    )

    ## ROW 5- MNO
    plot_performance_by_pro_anti_over_days(
        animal_trials_df,
        without_give=False,
        ax=ax_dict["M"],
        title=f"Pro: {animal_trials_df.pro_stim_set.dropna().unique()[0]},Anti: {animal_trials_df.anti_stim_set.dropna().unique()[0]}",
        xaxis_label=not (plot_give_del_info),
    )

    # TODO logic here for if switching is happning over days or sessions
    plot_n_pro_anti_blocks_days(
        animal_trials_df,
        ax_dict["N"],
        title="N Pro Anti Blocks",
        xaxis_label=False,
    )
    plot_cpoke_dur_timings_pregnp(
        animal_trials_df, ax_dict["O"], title="Cpoke Dur", xaxis_label=False
    )

    ## ROW 6- PQR

    if plot_give_del_info:
        plot_give_delay_dur_days(
            animal_trials_df,
            ax=ax_dict["P"],
            title="Anti Give Delay Dur",
            xaxis_label=False,
        )

    plot_block_switch_params(
        animal_trials_df, ax_dict["Q"], title="Block Switch Params", xaxis_label=False
    )

    plot_performance_by_give(
        animal_trials_df, ax_dict["R"], title="Perf by Give", xaxis_label=False
    )

    ## ROW 7- STU
    if plot_give_del_info:
        plot_give_use_rate_days(
            animal_trials_df,
            ax_dict["S"],
            title=f"Anti Give Use Rate, $\\alpha_-$= {animal_trials_df.give_del_adagrow_alpha_minus.iloc[-1]}",
            xaxis_label=True,
        )
    plot_give_type_and_block_switch_days(
        animal_trials_df,
        ax_dict["T"],
        title="Block Switch & Give Type",
        xaxis_label=True,
    )
    plot_rig_tech(animal_days_df, ax_dict["U"], title="Rig Tech", xaxis_label=True)

    return None
