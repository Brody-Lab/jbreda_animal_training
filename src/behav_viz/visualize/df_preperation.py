"""
Author: Jess Breda
Date: July 22, 2024
Description: functions to prepare DataFrames for plotting
for plots that are general and can be used across protocols
"""

import pandas as pd


def rename_give_types(df):
    """
    Rename the give types in the dataframe to be
    shorter & more readable for plotting

    params:
    -------
    df: pandas.DataFrame
        dataframe with a column `give_type_imp` that
        contains the give type for a trial

    returns:
    --------
    pandas.Series
        series with the give types renamed
    """
    assert "give_type_imp" in df.columns, "give_type_imp not in df"
    mapping = {"water_and_light": "w + l", "water": "w", "light": "l", "none": "n"}
    return df.give_type_imp.replace(mapping)


def rename_curricula(df):
    """
    Truncate curriculum names for plotting
    params:
    -------
    df: pandas.DataFrame
        dataframe with a column `curriculum` that
        contains the curriculum name for a trial

    returns:
    --------
    pandas.Series
        series with the curriculum renamed
    """

    assert "curriculum" in df.columns, "curriculum not in df"
    # remove "JB_" prefix
    return df.curriculum.str.replace("JB_", "")
