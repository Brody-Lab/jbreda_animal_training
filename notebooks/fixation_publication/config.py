""""
Config File with names, dates, notes for the fixation experiment
alpha cohort that can easily be imported into other scripts.
"""

from datetime import datetime
import pandas as pd
import seaborn as sns

# ANIMAL IDS

ANIMALS = [
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

GROUP_1_ANIMALS = [  # animals that started on the 20th
    "R040",
    "R041",
    "R042",
    "R043",
    "R044",
    "R045",
    "R046",
    "R047",
]

GROUP_2_ANIMALS = [
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

V1_ANIMALS = [  # animals in V1 growth fixation
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

V2_ANIMALS = [  # animals in V2 growth fixation
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


## DATES

START_DATES = {"group_1": "2024-07-20", "group_2": "2024-07-27"}
DATE_MIN = START_DATES["group_1"]
DATE_MAX = "2024-10-17"


def get_start_date(group, type="datetime"):
    """Returns the start date of the cohort
    subgroup (group 1, group 2 etc)
    as either a string or datetime object."""

    start_date = START_DATES[group]
    if type == "datetime":
        return datetime.strptime(start_date, "%Y-%m-%d")
    return start_date


DATE_DROPS = {
    # note that R044 had a rig break and needs to be dropped from that point forward and due to the large amount of dates, that is done in the ingest script
    "R042": pd.Timestamp(
        "2024-07-29"
    ).date(),  # stage 5 wasn't ready yet but animal was
    "R043": [
        pd.Timestamp("2024-07-26").date(),
        pd.Timestamp("2024-07-27").date(),
        pd.Timestamp("2024-07-28").date(),
        pd.Timestamp("2024-07-29").date(),  # stage 5 wasn't ready yet but animal was
    ],
    "R046": pd.Timestamp(
        "2024-08-12"
    ).date(),  # 2 trials shy of EOD on day before, should have manually moved into stage 9
    "R052": pd.Timestamp(
        "2024-08-14"
    ).date(),  # rig restart set fixation duration to 0.01 and restarted stage
    "R051": [
        pd.Timestamp("2024-08-20").date(),
        pd.Timestamp("2024-08-21").date(),
        pd.Timestamp("2024-08-22").date(),
    ],  # accidental extra days in stage 9 after probe complete
}

DROP_COLUMNS = [
    "n_incorr_spokes_during_give_del",
    "sa",
    "sb",
    "stimuli_on",
    "pre_dur",
    "adj_pre_dur",
    "stimulus_dur",
    "post_dur",
    "sb_extra_dur",
    "give_delay_dur",
    "give_xtra_light_delay_dur",
    "give_use",
    "give_del_growth_type",
    "give_del_adagrow_trial_subset",
    "give_del_adagrow_alpha_plus",
    "give_del_adagrow_alpha_minus",
    "give_del_adagrow_threshold",
    "give_del_adagrow_subset_prev_perf",
    "give_del_adagrow_step_size",
    "give_del_adagrow_window_size",
    "stim_set",
    "block_switch_type",
    "give_delay_strict",
    "volume_multiplier",
    "sound_pair",
]


## Aesthetic parameters
## Aesthetic parameters
HUE_ORDER_ANIMALS = V1_ANIMALS + V2_ANIMALS
HUE_ORDER_EXP = ["V1", "V2"]
V1_COLOR = "#9C1D4F"
V2_COLOR = "#1D9C6A"
EXP_PALETTE = [V1_COLOR, V2_COLOR]
# Generate the color palettes for each group using seaborn
V1_PALETTE = sns.color_palette("flare", len(V1_ANIMALS))
V2_PALETTE = sns.color_palette("crest", len(V2_ANIMALS))


# Create a lookup dictionary mapping each animal to its color
ANIMAL_PALETTE = {}
for animal, color in zip(V1_ANIMALS, V1_PALETTE):
    ANIMAL_PALETTE[animal] = color
for animal, color in zip(V2_ANIMALS, V2_PALETTE):
    ANIMAL_PALETTE[animal] = color


## Stages

PROBE_STAGES = [9, 10]
GROWING_STAGES = [5,6,7]
SPOKE_STAGES = [1,2,3,4]