""""
Config File with names, dates, notes for the alpha experiment
that can easily be imported into other scripts.
"""

from datetime import datetime
import pandas as pd

# ANIMAL IDS

ALPHA_ANIMALS = [
    "R040",
    "R041",
    "R042",
    "R043",
    "R044",
    "R045",
    "R046",
    "R047",
    "R048",
    "R049",
    "R050",
    "R051",
    "R052",
    "R053",
    "R054",
    "R055",
    "R056",
    "R057",
]

ALPHA_1_ANIMALS = [
    "R040",
    "R041",
    "R042",
    "R043",
    "R044",
    "R045",
    "R046",
    "R047",
]

ALPHA_2_ANIMALS = [
    "R048",
    "R049",
    "R050",
    "R051",
    "R052",
    "R053",
    "R054",
    "R055",
    "R056",
    "R057",
]

ALPHA_V1 = [
    "R040",
    "R042",
    "R044",
    "R046",
    "R048",
    "R050",
    "R052",
    "R054",
    "R056",
]

ALPHA_V2 = [
    "R041",
    "R043",
    "R045",
    "R047",
    "R049",
    "R051",
    "R053",
    "R055",
    "R057",
]


ALPHA_SESSION_6 = [
    "R040",
    "R041",
    "R042",
    "R043",
    "R044",
    "R045",
    "R046",
    "R047",
    "R048",
    "R049",
    "R050",
    "R051",
]

ALPHA_SESSION_7 = [
    "R052",
    "R053",
    "R054",
    "R055",
    "R056",
    "R057",
]
## DATES

ALPHA_START_DATES = {"alpha_1": "2024-07-20", "alpha_2": "2024-07-27"}


def get_start_date(group, type="datetime"):
    """Returns the start date of the cohort
    subgroup (alpha 1, alpha 2 etc)
    as either a string or datetime object."""

    start_date = ALPHA_START_DATES[group]
    if type == "datetime":
        return datetime.strptime(start_date, "%Y-%m-%d")
    return start_date


ALPHA_DATE_DROPS = {
    "R046": pd.Timestamp(
        "2024-08-12"
    ).date(),  # 2 trials shy of EOD on day before, should have manually moved into stage 9
    "R052": pd.Timestamp(
        "2024-08-14"
    ).date(),  # rig restart set fixation duration to 0.01 and restarted stage
    "R051": pd.Timestamp("2024-08-21").date(),  # accidental extra days in stage 9
    "R051": pd.Timestamp("2024-08-22").date(),  # accidental extra days in stage 9
}
