# src/behav_viz/ingest/__init__.py

from ..utils import dir_utils, dj_utils
from . import create_days_df, create_trials_df
import pandas as pd


def drop_dates(df, date_drop_dict):

    for animal_id, dates in date_drop_dict.items():
        # Ensure 'dates' is iterable, whether it's a single date or a list of dates
        if not isinstance(dates, list):
            dates = [dates]

        for date in dates:
            print("dropping", animal_id, date, "length:", len(df))
            df = df.query("not (animal_id == @animal_id and date == @date)")
            print("length:", len(df))

    # this animal had a rig break and needs to be dropped from that point forward
    r044_max_date = pd.Timestamp("2024-08-18").date()
    df = df.query("not (animal_id == 'R044' and date > @r044_max_date)")

    return df
