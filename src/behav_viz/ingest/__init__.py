# src/behav_viz/ingest/__init__.py

from ..utils import dir_utils, dj_utils
from . import create_days_df, create_trials_df


def drop_dates(df, date_drop_dict):

    for animal_id, date in date_drop_dict.items():
        print("dropping", animal_id, date, "length:", len(df))
        df = df.query("not (animal_id == @animal_id and date == @date)")
        print("length:", len(df))

    return df
