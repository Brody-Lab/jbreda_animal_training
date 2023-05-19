import datajoint as dj
import numpy as np
import pandas as pd
import datetime
from datajoint.errors import DataJointError
from dj_utils import ANIMAL_IDS


ratinfo = dj.create_virtual_module("intfo", "ratinfo")
bdata = dj.create_virtual_module("bdata", "bdata")


def fetch_daily_summary_info(
    animal_ids=None, date_min="2000-01-01", date_max="2030-01-01", verbose=False
):
    """
    TODO- PRIMARY FUNCTION
    TODO- add in overwrite/save out functions due to slow load in
    """
    animal_ids = ANIMAL_IDS if animal_ids is None else animal_ids
    assert type(animal_ids) == list, "animal ids must be in a list"

    animals_daily_summary_df = []

    for animal_id in animal_ids:
        subject_session_key = {"ratname": animal_id}
        date_min_key = f"sessiondate >= '{date_min}'"
        date_max_key = f"sessiondate <= '{date_max}'"

        # get dates where there are entries to sessions table
        dates = (
            bdata.Sessions & subject_session_key & date_min_key & date_max_key
        ).fetch("sessiondate")

        dates = np.unique(dates)  # drop repeats

        animals_daily_summary_df.append(crate_animal_daily_summary_df(animal_id, dates))
    daily_summary_df = pd.concat(animals_daily_summary_df)
    return daily_summary_df


def crate_animal_daily_summary_df(animal_id, dates):
    """
    TODO
    """
    session_dfs = []
    water_mass_dfs = []

    for date in dates:
        session_dfs.append(fetch_daily_session_info(animal_id, date))
        water_mass_dfs.append(fetch_daily_water_and_mass_info(animal_id, date))
    daily_summary_df = pd.merge(
        pd.concat(session_dfs, ignore_index=True),
        pd.concat(water_mass_dfs, ignore_index=True),
        on=["animal_id", "date"],
    )

    print(
        f"fetched {len(dates)} daily summaries for {animal_id} between {min(dates)} and {max(dates)}"
    )

    return daily_summary_df


########################
### SESSION INFO FXs ###
########################


def fetch_daily_session_info(animal_id, date):
    """
    TODO
    """

    # fetch data
    query_keys = {
        "ratname": animal_id,
        "sessiondate": date,
    }  # specific to Sessions table

    n_done_trials, rigid, start_times, end_times = (
        bdata.Sessions & query_keys & "n_done_trials > 1"
    ).fetch("n_done_trials", "hostname", "starttime", "endtime")

    # create dict
    D = {}

    D["animal_id"] = animal_id
    D["date"] = date
    D["rigid"] = rigid[-1]
    D["n_done_trials"] = np.sum(n_done_trials)
    D["n_sessions"] = len(n_done_trials)

    st = start_times.min()
    D["start_time"] = datetime.datetime.strptime(str(st), "%H:%M:%S").time()

    D["train_dur_hrs"] = calculate_daily_train_dur(
        start_times, end_times, units="hours"
    )
    D["trial_rate"] = np.round(D["n_done_trials"] / D["train_dur_hrs"], decimals=2)

    return pd.DataFrame(D, index=[0])


###  SUB FUNCTIONS  ###


def calculate_daily_train_dur(start_times, end_times, units="hours"):
    """
    function that calculates total train time given
    start and end times for a given animal, day

    inputs
    ------
    start_times : arr
        array of start times as datetime.timedelta objects from sessions
        table for a given animal, day. Typically of len == 1
    end_times : arr
        array of start times as datetime.timedelta objects from sessions
        table for a given animal, day. Typically of len == 1
    units : str, "hours" (default), "minutes" or "seconds
        what units to return trial duration

    returns
    ------
    daily_train_dur : float
        amount of time (in specified "units") a of training for given
        animal, day
    """

    if units == "hours":
        time_conversion = 3600
    elif units == "minutes":
        time_conversion = 60
    elif units == "seconds":
        time_conversion = 1

    daily_train_dur_seconds = np.sum(end_times - start_times)

    daily_train_dur = daily_train_dur_seconds.total_seconds() / time_conversion

    return np.round(daily_train_dur, decimals=2)


########################
### WATER & MASS FXs ###
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
    D["mass"], D["tech"] = fetch_daily_mass(animal_id, date)
    D["percent_target"] = fetch_daily_restriction_target(animal_id, date)
    D["pub_volume"] = fetch_pub_volume(animal_id, date)
    D["rig_volume"] = fetch_rig_volume(animal_id, date)
    D["volume_target"] = fetch_daily_water_target(
        D["mass"], D["percent_target"], verbose=False
    )
    D["water_diff"] = (D["pub_volume"] + D["rig_volume"]) - D["volume_target"]

    return pd.DataFrame(D, index=[0])


###  SUB FUNCTIONS  ###


def fetch_daily_water_target(mass, percent_target, verbose=False):
    """
    Function for getting an animals water volume target on
    a specific date

    inputs
    ------
    mass : float
        mass in grams for a given animal, date fetched the Mass table
        using fetch_daily_mass()
    percent_target : float
        water restriction target in terms of percentage of body weight
        fetched from the Water table using fetch_daily_restriction_target()
    verbose : bool
        if you want to print restriction information

    returns
    ------
    volume_target : float
        water restriction target in mL
    """
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
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    returns
    ------
    mass : float
        weight in grams on date
    tech : str
        initials of technician that weighed given animal, date
    """

    Mass_keys = {"ratname": animal_id, "date": date}
    try:
        mass, tech = (ratinfo.Mass & Mass_keys).fetch1("mass", "tech")
    except DataJointError:
        print(
            f"mass data not found for {animal_id} on {date},",
            f"using previous days mass",
        )
        prev_date = date - datetime.timedelta(days=1)
        Mass_keys = {"ratname": animal_id, "date": prev_date}
        mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))
        tech = "NA"
    return float(mass), tech


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
