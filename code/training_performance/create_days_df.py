"""
Author: Jess Breda
Date: May 31, 2023
Description: Code to create a day-level summary dataframe from
various DataJoint tables to be used in analyzing animal health
and performance over time.
"""
import datajoint as dj
import numpy as np
import pandas as pd
import datetime
from datajoint.errors import DataJointError
from pathlib import Path
import os


ratinfo = dj.create_virtual_module("intfo", "ratinfo")
bdata = dj.create_virtual_module("bdata", "bdata")

#############

#############


def create_days_df_from_dj(
    animal_ids, date_min="2000-01-01", date_max="2030-01-01", verbose=False
):
    """
    Function to create a day level summary df from DataJoint
    tables. This function is a wrapper for the following
    functions:
        - create_animal_daily_summary_df

    params
    ------
    animal_ids : list (optional, default = None)
        list of animal ids to query.
    date_min : str (optional, default = "2000-01-01")
        minimum date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    date_max : str (optional, default = "2030-01-01")
        maximum date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    verbose : bool (optional, default = False)
        whether to print out verbose statements

    returns
    -------
    daily_summary_df : pd.DataFrame
        data frame containing day-level summary info for all animals
        in `animal_ids` between `date_min` and `date_max`
    """
    assert type(animal_ids) == list, "animal ids must be in a list"

    animals_daily_summary_df = []

    for animal_id in animal_ids:
        # create keys for querying dj tables to determine which dates
        # to fetch
        subject_key = {"ratname": animal_id}
        sess_date_min_key = f"sessiondate >= '{date_min}'"
        sess_date_max_key = f"sessiondate <= '{date_max}'"
        mass_date_min_key = f"date >= '{date_min}'"
        mass_date_max_key = f"date <= '{date_max}'"

        # get dates where there are entries to sessions or mass table
        sess_dates = (
            bdata.Sessions & subject_key & sess_date_min_key & sess_date_max_key
        ).fetch("sessiondate")

        mass_dates = (
            ratinfo.Mass & subject_key & mass_date_min_key & mass_date_max_key
        ).fetch("date")

        dates = np.unique(np.concatenate((mass_dates, sess_dates)))  # drop repeats

        # create df for given dates for an animal via dj fetch & formatting
        animals_daily_summary_df.append(
            create_animal_days_summary_df(animal_id, dates, verbose=verbose)
        )

    # concatenate over animals
    daily_summary_df = pd.concat(animals_daily_summary_df)
    return daily_summary_df


def create_animal_days_summary_df(animal_id, dates, verbose=False):
    """
    Function to create a day level summary df from DataJoint
    tables for a given animal and dates. This function is a wrapper
    for the following functions:
        - fetch_daily_session_info
        - fetch_daily_water_and_mass_info

    params
    ------
    animal_id : str,
        animal name e.g. "R501"
    dates : list
        list of dates to query in YYYY-MM-DD format, e.g. "2022-01-04"
        that an animal had entry in Session or Mass table
    verbose : bool (optional, default = False)
        whether to print out verbose statements

    returns
    -------
    daily_summary_df : pd.DataFrame
        data frame containing daily summary info for a given animal
        and dates
    """
    session_dfs = []
    water_mass_dfs = []

    for date in dates:
        session_dfs.append(fetch_day_session_info(animal_id, date))
        water_mass_dfs.append(
            fetch_day_water_and_mass_info(animal_id, date, verbose=verbose)
        )
    daily_summary_df = pd.merge(
        pd.concat(session_dfs, ignore_index=True),
        pd.concat(water_mass_dfs, ignore_index=True),
        on=["animal_id", "date"],
    )
    if verbose:
        print(
            f"\nfetched {len(dates)} daily summaries for {animal_id} "
            f"from dj between {min(dates)} and {max(dates)}"
        )

    return daily_summary_df


########################
### SESSION INFO FXs ###
########################


def fetch_day_session_info(animal_id, date):
    """
    Function to generate a df row containing session info
    for a given animal, date

    params
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"

    returns
    -------
    D : pd.DataFrame
        data frame row with summary session info for a
        given animal, date
    """

    # fetch data
    query_keys = {
        "ratname": animal_id,
        "sessiondate": date,
    }  # specific to Sessions table

    (
        n_done_trials,
        rigid,
        start_times,
        end_times,
        *perf_metrics,
    ) = (bdata.Sessions & query_keys & "n_done_trials > 1").fetch(
        "n_done_trials",
        "hostname",
        "starttime",
        "endtime",
        "total_correct",
        "percent_violations",
        "right_correct",
        "left_correct",
    )

    # create dict
    D = {}

    # no session for this day, animal only weighed
    if len(n_done_trials) == 0:
        D["animal_id"] = animal_id
        D["date"] = date
        D["rigid"] = np.nan
        D["n_done_trials"] = np.nan
        D["n_sessions"] = 0
        D["start_time"] = np.nan
        D["train_dur_hrs"] = 0
        D["trial_rate"] = np.nan
        D["hit_rate"] = np.nan
        D["viol_rate"] = np.nan
        D["side_bias"] = np.nan

    else:
        D["animal_id"] = animal_id
        D["date"] = date
        D["rigid"] = rigid[-1]
        D["n_done_trials"] = np.sum(n_done_trials)
        D["n_sessions"] = len(n_done_trials)

        st = start_times.min()
        D["start_time"] = datetime.datetime.strptime(str(st), "%H:%M:%S").time()

        D["train_dur_hrs"] = calculate_day_train_dur(
            start_times, end_times, units="hours"
        )
        D["trial_rate"] = np.round(D["n_done_trials"] / D["train_dur_hrs"], decimals=2)

        D["hit_rate"], D["viol_rate"], D["side_bias"] = calculate_perf_metrics(
            perf_metrics, n_done_trials
        )

    return pd.DataFrame(D, index=[0])


###  SUB FUNCTIONS  ###


def calculate_day_train_dur(start_times, end_times, units="hours"):
    """
    Function to calculate the amount of time an animal trained
    for a given day

    params
    ------
    start_times : arr
        array of start times as datetime.timedelta objects from sessions
        table for a given animal, day. Typically of len == 1.
    end_times : arr
        array of end times as datetime.timedelta objects from sessions
        table for a given animal, day. Typically of len == 1.
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


def calculate_perf_metrics(perf_metrics, n_done_trials):
    """
    Function to calculate weighted averages of Session
    table performance metrics given the number of trials
    from multiple session in the same day.

    params
    ------
    perf_metrics: list
        List of performance metrics from Sessions table to calculate
        weighted average of. Typically of len == 4. Order is:
        total_correct, percent_violations, right_correct, left_correct
    n_done_trials: list
        N_done_trials from Sessions table. Must be the same length
        as perf_metrics.

    returns
    -------
    hit_rate : float
        Weighted average of hit rate
    viol_rate : float
        Weighted average of violation rate
    side_bias : float
        Weighted average of side bias (- = left, + = right)
    """

    # calculate weighted average of performance metrics
    hit_rate = np.average(perf_metrics[0], weights=n_done_trials)
    viol_rate = np.average(perf_metrics[1], weights=n_done_trials)
    side_bias = np.average(perf_metrics[2] - perf_metrics[3], weights=n_done_trials)

    return hit_rate, viol_rate, side_bias


########################
### WATER & MASS FXs ###
########################


def fetch_day_water_and_mass_info(animal_id, date, verbose=False):
    """
    Function to generate a df row containing mass,
    water and restriction data for a given animal, date

    params
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    verbose : bool (optional, default = False)
        whether to print out verbose statements

    returns
    -------
    D : pd.DataFrame
        data frame row with mass, restriction and water data
        for a given animal, date
    """
    D = {}

    D["animal_id"] = animal_id
    D["date"] = date
    D["mass"], D["tech"] = fetch_day_mass(animal_id, date, verbose=verbose)
    D["percent_target"] = fetch_day_restriction_target(animal_id, date)
    D["pub_volume"] = fetch_pub_volume(animal_id, date)
    D["rig_volume"] = fetch_rig_volume(animal_id, date, verbose=verbose)
    D["volume_target"] = fetch_day_water_target(
        D["mass"], D["percent_target"], D["date"], verbose=verbose
    )
    D["water_diff"] = (D["pub_volume"] + D["rig_volume"]) - D["volume_target"]

    return pd.DataFrame(D, index=[0])


###  SUB FUNCTIONS  ###


def fetch_day_water_target(mass, percent_target, date, verbose=False):
    """
    Function to calculate the water restriction target
    for a given animal, date

    params
    ------
    mass : float
        mass in grams for a given animal, date fetched from
        the Mass table using fetch_daily_mass()
    percent_target : float
        water restriction target in terms of percentage of
        body weight fetched from the Water table using
        fetch_daily_restriction_target()
    date : str or datetime
        date queried in YYYY-MM-DD format, e.g. "2022-01-04"

    returns
    ------
    volume_target : float
        water restriction target in mL
    """
    # sometimes the pub isn't run- let's assume the minimum value
    if percent_target == 0:
        percent_target = 4 if mass < 100 else 3
        if verbose:
            print(f"Percent target was empty on {date}, defaulting to minimum.")

    volume_target = np.round((percent_target / 100) * mass, 2)

    return volume_target


def fetch_day_mass(animal_id, date, verbose=False):
    """
    Function for getting an animals mass on
    a specific date

    params
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    verbose : bool (optional, default = False)
        whether to print out verbose statements

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
        if verbose:
            print(
                f"mass data not found for {animal_id} on {date}, but animal trained",
                f"using previous days mass",
            )
        prev_date = date - datetime.timedelta(days=1)
        Mass_keys = {"ratname": animal_id, "date": prev_date}
        mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))
        tech = np.nan
    return float(mass), tech


def fetch_day_restriction_target(animal_id, date):
    """
    Function for getting an animals water
    target for a specific date

    params
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
    percent_target = (ratinfo.Water & Water_keys).fetch("percent_target")

    if len(percent_target) == 0:
        # !!NOTE assumption made here will default to 3% for rats and 4% for mice
        # !! in the fetch_daily_water_target() function
        percent_target = 0
    elif len(percent_target) > 1:
        percent_target = percent_target.max()

    return float(percent_target)


def fetch_rig_volume(animal_id, date, verbose=False):
    """ "
    Fetch rig volume from RigWater table

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04"
    verbose : bool (optional, default = False)
        whether to print out verbose statements

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
        if verbose:
            print(f"rig volume was empty on {date}, defaulting to 0 mL")

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


def fetch_and_format_single_day_water(animal_id, date, verbose=False):
    """
    wrapper function to fetch mass and watering info for a single day
    and return the information in a dataframe for easy plotting
    wrt plot_trials_info/plot_watering_amounts()

    params
    ------
    animal_id: str, e.g. 'R610'
        id of animal of which to fetch data for from mass and water tables
    date: str, e.g. '2021-04-15'
        date of which to fetch data for from mass and water tables
    verbose: bool, default False
        whether to print out information about the animal's mass and water
        consumption

    returns
    -------
    df: pd.DataFrame
        dataframe with columns 'date', 'rig_volume', 'pub_volume' to make
        a stacked bar chart with (this is the preferred format for plotting)
    volume_target: float
        target volume of water to be consumed by the animal on given day to
        mark on plot (i.e. the minimum threshold)
    """
    mass, _ = fetch_day_mass(animal_id, date)
    percent_target = fetch_day_restriction_target(animal_id, date)
    pub_volume = fetch_pub_volume(animal_id, date)
    rig_volume = fetch_rig_volume(animal_id, date, verbose=verbose)
    volume_target = fetch_day_water_target(mass, percent_target, date, verbose=verbose)

    df = pd.DataFrame(
        {"date": [date], "rig_volume": [rig_volume], "pub_volume": [float(pub_volume)]}
    )
    return df, volume_target


######################
### LAZY LOAD DATA ###
######################


def lazy_load_days_summary_df(
    date_min,
    date_max,
    animal_ids,
    save_dir=os.getcwd(),
    f_name="days_df.csv",
    save_out=False,
    verbose=False,
):
    """
    Function to load in a pre-saved daily summary table and append new dates if needed.
    If no pre-saved df is found, then the function will fetch the data from DataJoint.

    params
    ------
    date_min : str
        start date in format "YYYY-MM-DD"
    date_max : str
        end date in format "YYYY-MM-DD"
    animal_ids : list
        list of animal ids to fetch data for, e.g. ["R610", "W600"]
    save_dir : str (optional, default = os.getcwd())
        directory to look for the pre-saved df, also used for save out
    f_name : str (optional, default = "summary_table.csv")
        name of the pre-saved df, also used for save out
    save_out : bool (optional, default = False)
        if any new info was fetched from dj, whether to save out. if
        appending occurred, this will overwrite the pre-saved df
    verbose : bool (optional, default = False)
        whether to print out verbose statements in dj fetch functions

    returns
    -------
    pandas.DataFrame
        daily summary table with specified animal ids in range [date_min, date_max]
    """
    full_path = Path(save_dir).joinpath(f_name)

    if full_path.exists():
        # load in pre-saved df
        pre_saved_df = pd.read_csv(full_path)

        # check that all animals are in the pre-saved df
        excluded_animals = np.setdiff1d(animal_ids, pre_saved_df.animal_id.unique())
        assert len(excluded_animals) == 0, (
            f"Pre-saved df does not contain {excluded_animals}! "
            "Overwrite functionality assumes all animals are in the DataFrame."
        )

        # check if there are any new dates to load
        new_dates = pd.date_range(start=date_min, end=date_max).difference(
            pd.date_range(start=pre_saved_df.date.min(), end=pre_saved_df.date.max())
        )

        # no new dates to load, append the previously saved df
        if not len(new_dates):
            print(
                f"Loaded pre-saved df with entries between {date_min} and {date_max}."
            )
            return pre_saved_df.query("date >=@date_min and date <=@date_max")

        # there are new dates to fetch, need to update or min and max dates
        new_min = new_dates.min().strftime("%Y-%m-%d")  # min date not in pre-saved df
        new_max = new_dates.max().strftime("%Y-%m-%d")  # max date not in pre-saved df

        if new_max < date_max:
            print(f"partial dj load with new date max {date_max} -> {new_max}")
            pre_saved_df = pre_saved_df.query("date <=@date_max")
            date_max = new_max
        elif new_min > date_min:
            print(f"partial dj load with new date min {date_min} -> {new_min}")
            pre_saved_df = pre_saved_df.query("date >=@date_min")
            date_min = new_min
        else:
            print(
                f"The provided date window is larger than the whole DataFrame. "
                f"Only one-sided lazy loading can be performed. "
                f"\nReturning the pre-saved DataFrame. "
                f"with dates between {pre_saved_df.date.min()} and {pre_saved_df.date.max()}."
            )
            return pre_saved_df
    else:
        print(f"No pre-saved df found, fetching from DataJoint.")

    # load in new data
    dj_df = create_days_df_from_dj(
        animal_ids=animal_ids, date_min=date_min, date_max=date_max, verbose=verbose
    )

    # append & return if we have a pre-saved df
    if "pre_saved_df" in locals() and isinstance(pre_saved_df, pd.DataFrame):
        appended_df = pd.concat([pre_saved_df, dj_df], ignore_index=True)
        appended_df["date"] = pd.to_datetime(appended_df["date"]).dt.date
        print(
            f"Returning appended df with entries between {appended_df.date.min()} and {appended_df.date.max()}"
        )

        if save_out:
            appended_df.to_csv(full_path, index=False)
            print(f"Saved out appended df to {full_path}")

        return appended_df.sort_values(by=["animal_id", "date"]).reset_index(drop=True)

    # otherwise return the dj_df
    else:
        if save_out:
            dj_df.to_csv(full_path, index=False)
            print(f"Saved out dj df to {full_path}")
        print(f"Returning dj df with entries between {date_min} and {date_max}")
        return dj_df
