"""
Author: Jess Breda
Date: July 22, 2024
Description: given days, trials table info make summary plots
for an animal for over days. Specifically for Fixation Grower protocol.
"""

import matplotlib.pyplot as plt

from behav_viz.visualize.plot_days_info import *
from behav_viz.visualize.plot_trials_info import *
from behav_viz.visualize.FixationGrower.plots import *
from behav_viz.visualize.plots import *


# Next step - get a multi plot frame work in here as outlined in git and then
# add the new raw/relative fix dur plot to the multiplot with other made plots
# then, done for the day

######################################################################################
#########                        SINGLE DAY PLOTS                            #########
######################################################################################


######################################################################################
#########                        MULTI DAY PLOTS                             #########
######################################################################################


def over_days_summary(animal_id, animal_days_df, animal_trials_df):
    """
    TODO make this into a wrapper!

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
    letters = sorted(list(set(layout.replace("\n", "").replace(" ", ""))))
    bar_plots = list("CEIL")  # manual input axes with bar plots
    bottom_row = letters[-3:]

    fig = plt.figure(constrained_layout=True, figsize=(30, 20))
    ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
    plt.suptitle(f"{animal_id} Daily Summary Plot", fontweight="semibold")

    # determine training stage info
    current_stage = animal_trials_df.stage.iloc[-1]
    current_sma = animal_trials_df.SMA_set.iloc[-1]

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


def over_days_summary_spoke(animal_id, animal_days_df, animal_trials_df):
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

    # TODO
    1. Run time
    2. Inter Trial Stats


    """

    layout = """
    AAABBBCCC
    DDDEEEFFF
    GGGHHHIII
    JJJKKKLLL
    MMMNNNOOO
    """
    letters = sorted(list(set(layout.replace("\n", "").replace(" ", ""))))
    bar_plots = list("CFIL")  # manual input axes with bar plots
    bottom_row = letters[-3:]

    exp_condition = animal_trials_df.fix_experiment.iloc[-1]

    fig = plt.figure(constrained_layout=True, figsize=(30, 20))
    ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
    plt.suptitle(
        f"{exp_condition} Condition | {animal_id} Daily Summary Plot",
        fontweight="semibold",
    )

    ## ROW 1- rig/techsession/foodpuck -- mass -- water
    plot_stage(animal_trials_df, ax_dict["A"], title="Stage")
    plot_mass(animal_days_df, ax_dict["B"], title="Mass")
    plot_water_restriction(animal_days_df, ax_dict["C"], title="Water", legend=False)

    ## ROW 2- stage -- run time --trials
    plot_rig_tech_foodpuck(animal_days_df, ax_dict["D"], title="Tech & Rig Info")
    plot_trials(animal_days_df, ax_dict["E"], title="Trials", legend=True)
    # TODO plot_run_time(animal_days_df, ax_dict["F"], title="Run Time")
    ax_dict["F"].set_title("Run Time")

    ## ROW 3- perf -- perf bars -- ?
    plot_curriculum_and_give_types(
        animal_trials_df, ax_dict["G"], title="Curriculum & Give Types"
    )
    plot_performance(animal_days_df, ax_dict["H"], title="Performance")
    plot_performance_bars(animal_trials_df, ax_dict["I"], title="Performance")

    ## ROW 4 - side bias -- antibias -- side bias params
    plot_side_bias(animal_days_df, ax_dict["J"], title="Side Bias")
    plot_antibias_probs(animal_trials_df, ax_dict["K"], title="Antibias")
    plot_sidebias_params(animal_trials_df, ax_dict["L"], title="Side Bias Params")

    ## ROW 5 - inter trial stats -- time to spoke - give type/curriculum/condition
    # TODO plot_inter_trial_states(animal_trials_df, ax_dict["M"], title="Inter Trial Stats")
    ax_dict["M"].set_title("Inter Trial Stats")
    plot_time_to_spoke(animal_trials_df, ax_dict["N"], title="Time to Spoke")

    pu.adjust_mosaic_axes(ax_dict, letters, bar_plots, bottom_row, animal_days_df)


def over_days_summary_cpoke_learning(animal_id, animal_days_df, animal_trials_df):
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

    # TODO
    2. Run Time plot
    4. Inter Trial Stats
    5. Trial Structure for FG
    6. Cpoke Failure Rates
    7. N Settling Ins
    8. Min/Max Deltas
    9. Fix Dur Delta Over Days


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
    letters = sorted(list(set(layout.replace("\n", "").replace(" ", ""))))
    bar_plots = list("CFILO")  # manual input axes with bar plots
    bottom_row = letters[-3:]

    exp_condition = animal_trials_df.fix_experiment.iloc[-1]

    fig = plt.figure(constrained_layout=True, figsize=(30, 25))
    ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
    plt.suptitle(
        f"{exp_condition} Condition | {animal_id} Daily Summary Plot",
        fontweight="semibold",
    )

    ## ROW 1- rig/techsession/foodpuck -- mass -- water
    plot_stage(animal_trials_df, ax_dict["A"], title="Stage")
    plot_mass(animal_days_df, ax_dict["B"], title="Mass")
    plot_water_restriction(animal_days_df, ax_dict["C"], title="Water", legend=False)

    ## ROW 2- stage -- run time --trials
    plot_rig_tech_foodpuck(animal_days_df, ax_dict["D"], title="Tech & Rig Info")
    plot_trials(animal_days_df, ax_dict["E"], title="Trials", legend=True)
    # TODO plot_run_time(animal_days_df, ax_dict["F"], title="Run Time")

    ## ROW 3- perf -- perf bars -- ?
    plot_curriculum_and_give_types(
        animal_trials_df, ax_dict["G"], title="Curriculum & Give Types"
    )
    plot_performance(animal_days_df, ax_dict["H"], title="Performance")
    plot_performance_bars(animal_trials_df, ax_dict["I"], title="Performance")

    ## ROW 4 - side bias -- antibias -- side bias params
    plot_side_bias(animal_days_df, ax_dict["J"], title="Side Bias")
    plot_antibias_probs(animal_trials_df, ax_dict["K"], title="Antibias")
    plot_sidebias_params(animal_trials_df, ax_dict["L"], title="Side Bias Params")

    ## ROW 5 - inter trial stats -- time to spoke - give type/curriculum/condition
    # TODO plot_inter_trial_states(animal_trials_df, ax_dict["M"], title="Inter Trial Stats")
    plot_time_to_spoke(animal_trials_df, ax_dict["N"], title="Time to Spoke")
    plot_trial_structure(animal_trials_df, ax_dict["O"], title="Trial Structure")

    ## ROW 6 - failure rates -- n settling ins/trial -- min/max deltas
    plot_cpoke_dur_timings_pregnp(
        animal_trials_df, ax_dict["P"], title="Cpoke Dur"
    )  # TODO remove

    plot_n_cpokes_and_multirate(
        animal_trials_df, ax_dict["Q"], title="Multi Cpokes"
    )  # TODO remove

    # TODO plot_cpoke_failure_rates(animal_trials_df, ax_dict["P"], title="Failure Rates")
    # TODO plot_n_settling_ins(animal_trials_df, ax_dict["Q"], title="N Settling Ins")
    # TODO plot_delta_fix_dur(animal_trials_df, ax_dict["R"], title="Min/Max Deltas")

    ## ROW 7 - cpoke dur raw -- cpoke dur relative -- delta over days
    plot_cpoke_fix_stats_raw(animal_trials_df, ax_dict["S"], title="Cpoke Fix Dur Raw")
    plot_cpoke_fix_stats_relative(
        animal_trials_df, ax_dict["T"], title="Cpoke Fix Dur Relative"
    )
    # TODO plot_fix_dur_delta_over_days(animal_trials_df, ax_dict["U"], title="Fix Dur Delta Over Days")

    pu.adjust_mosaic_axes(ax_dict, letters, bar_plots, bottom_row, animal_days_df)
