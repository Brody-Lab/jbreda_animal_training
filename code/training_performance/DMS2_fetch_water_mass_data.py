import datajoint as dj
import numpy as np
import pandas as pd
from datetime import timedelta
from datajoint.errors import DataJointError


ratinfo = dj.create_virtual_module("intfo", "ratinfo")

########################
### SUMMARY FUNCTION ###
########################


def fetch_daily_water_and_mass_info(animal_id, date):
    """ "
    Wrapper function to generate a df row containing mass,
    water and restriction data for a given animal, date

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"

    returns
    -------
    D : df
        data frame row with mass, restriction and water data
        for a given animal, date
    """
    D = {}

    D["animal_id"] = animal_id
    D["date"] = date
    D["pub_volume"] = fetch_pub_volume(animal_id, date)
    D["rig_volume"] = fetch_rig_volume(animal_id, date)
    D["volume_target"], D["percent_target"], D["mass"] = fetch_daily_water_target(
        animal_id, date, verbose=False, return_mass_and_percent=True
    )

    return pd.DataFrame(D, index=[0])


########################
###  SUB FUNCTIONS   ###
########################


def fetch_daily_water_target(
    animal_id, date, verbose=False, return_mass_and_percent=False
):
    """
    Function for getting an animals water volume target on
    a specific date

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04
    verbose : bool
        if you want to print restriction information
    return_mass_and_percent : bool
        if you want to return the volume_target, percent_target and
        mass together. useful for the building the sessions date table

    returns
    ------
    volume_target : float
        water restriction target in mL
    """

    percent_target = fetch_daily_restriction_target(animal_id, date)
    mass = fetch_daily_mass(animal_id, date)

    # sometimes the pub isn't run- let's assume the minimum value
    if percent_target == 0:
        percent_target = 4
        note = "Note set to 0 but assumed 4."
    else:
        note = ""

    volume_target = np.round((percent_target / 100) * mass, 2)

    if verbose:
        print(
            f"""On {date} {animal_id} is restricted to:
        {percent_target}% of body weight or {volume_target} mL
        {note}
        """
        )

    if return_mass_and_percent:
        return volume_target, percent_target, mass
    else:
        return volume_target


def fetch_daily_mass(animal_id, date):
    """
    Function for getting an animals mass on
    a specific date
    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04
    returns
    ------
    mass : float
        weight in grams on date
    """

    Mass_keys = {"ratname": animal_id, "date": date}
    try:
        mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))
    except:
        print(
            f"mass data not found for {animal_id} on {date},",
            f"using previous days mass",
        )
        prev_date = date - timedelta(days=1)
        Mass_keys = {"ratname": animal_id, "date": prev_date}
        mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))
    return mass


def fetch_daily_restriction_target(animal_id, date):
    """
    Function for getting an animals water
    target for a specific date
    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04
    returns
    ------
    percent_target : float
        water restriction target in terms of percentage of body weight
    note
    ----
    You can also fetch this code from the registry, but it's not
    specific to the date. See code below.
    ```
    # fetch from comments section e.g. 'Mouse Water Pub 4'
    r500_registry = (ratinfo.Rats & 'ratname = "R501"').fetch1()
    comments = r500_registry['comments']
    #split after the word pub,only allow one split and
    # grab whatever follows the split
    target = float(comments.split('Pub',1)[1])
    ```
    """

    Water_keys = {"rat": animal_id, "date": date}

    # can't do fetch1 with this because water table
    # sometimes has a 0 entry and actual entry so
    # I'm taking the max to get around this
    # this needs to be address w/ DJ people
    percent_target = (ratinfo.Water & Water_keys).fetch("percent_target")

    if len(percent_target) == 0:
        percent_target = 4  # NOTE assumption made here to 4%- be careful!
    elif len(percent_target) > 1:
        percent_target = percent_target.max()

    return float(percent_target)


def fetch_rig_volume(animal_id, date):
    """ "
    Fetch rig volume from RigWater table

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"

    returns
    -------
    rig_volume : float
        rig volume drunk in mL for a given animal, day
    """

    Rig_keys = {"ratname": animal_id, "dateval": date}  # Specific to Rigwater table
    try:
        rig_volume = float((ratinfo.Rigwater & Rig_keys).fetch1("totalvol"))
    except DataJointError:
        rig_volume = 0
        print(f"rig volume wasn't tracked on {date}, defaulting to 0 mL")

    return rig_volume  # note this doesn't account for give water as of 5/18/2023


def fetch_pub_volume(animal_id, date):
    """
    Fetch pub volume from Water table

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"

    returns
    -------
    pub_volume : float
        pub volume drunk in mL for a given animal, day
    """

    Water_keys = {"rat": animal_id, "date": date}  # specific to Water table
    pub_volume = (ratinfo.Water & Water_keys).fetch("volume")

    # pub volume doesn't always have 1 entry
    if len(pub_volume) == 0:
        pub_volume = 0
    elif len(pub_volume) > 1:
        pub_volume = pub_volume.max()

    return float(pub_volume)
