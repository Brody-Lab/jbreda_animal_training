"""
Author: Jess Breda
Date: July 12, 2023
Description: given days, trials table info make summary plots
for an animal for a given day or across multiple
"""

import matplotlib.pyplot as plt

from behav_viz.visualize.plot_days_info import *
from behav_viz.visualize.plot_trials_info import *

from behav_viz.visualize import DMS2
from behav_viz.visualize import FixationGrower


######################
###   SINGLE DAY   ###
######################


def single_day_summaries(trials_df, figures_path, save_out=True, overwrite=False):
    """
    function to plot summaries for each animal, day in a df of
    trials data. will handle lf

    trials_dfs : pd.DataFrame
        df of trials data loaded in using `create_trials_df_from_dj`
    figures_path : Path object
        path to save figures to
    overwrite : bool (optional, default = False)
        whether to overwrite existing figures with same name
    """

    for (_, _), sub_df in trials_df.groupby(["date", "animal_id"]):
        is_dms_protocol = "DMS" in trials_df.protocol.iloc[-1]
        is_fixation_protocol = "Fixation" in trials_df.protocol.iloc[-1]

        if is_dms_protocol:
            DMS2.multiplots.single_day_summary(
                sub_df, figures_path, save_out=save_out, overwrite=overwrite
            )
        elif is_fixation_protocol:
            DMS2.multiplots.single_day_summary(
                sub_df, figures_path, save_out=save_out, overwrite=overwrite
            )
        else:
            raise ValueError("Protocol not recognized")


######################
###   MULTIDAY     ###
######################


def over_days_summaries(animal_id, days_df, trials_df):
    for animal_id, animal_days_df in days_df.groupby("animal_id"):
        animal_trials_df = trials_df.query("animal_id == @animal_id")

        current_stage = animal_trials_df.stage.iloc[-1]
        is_dms_protocol = "DMS" in trials_df.protocol.iloc[-1]
        is_fixation_protocol = "Fixation" in trials_df.protocol.iloc[-1]

        if is_dms_protocol:
            if current_stage < 12:
                DMS2.multiplots.over_days_summary_pre_pro_anti(
                    animal_id, animal_days_df, animal_trials_df
                )
            else:
                DMS2.multiplots.over_days_summary_pro_anti(
                    animal_id, animal_days_df, animal_trials_df
                )

        if is_fixation_protocol:
            # TODO implement logic here! this is just a placeholder
            FixationGrower.multiplots.over_days_summary(
                animal_id, animal_days_df, animal_trials_df
            )
