import numpy as np
import datajoint as dj
import pandas as pd
import datetime

ratinfo = dj.create_virtual_module("intfo", "ratinfo")
bdata = dj.create_virtual_module("bdata", "bdata")

###################
###  FUNCTIONS  ###
###################


def return_date_window(latest_date=None, n_days_back=None):
    """
    Function to create a date window for querying the DataJoint
    SQL database. Pairs nicely with `fetch_latest_trials_data`
    or `fetch_daily_summary_data`

    params
    ------
    latest_date : str  (optional, default = None)
        latest date to include in the window, defaults to today
        if left empty
    n_days_back : int (optional, default = None)
        number of days back from `latest_date` to include,
        defaults to all days if left empty

    Note: if you are out of range of your table (e.g min date)
    is before the start of training) it's okay.

    returns
    ------
    min_date : str
        minimum date in the specified date window
    max_date : str
        maximum date in the specified date window

    example usage
    ------------
    `return_date_window(latest_date='2022-11-3', n_days_back=8)`
    `return_date_window(latest_date=None, n_days_back=7)`
    """

    if latest_date:
        date_max = pd.to_datetime(latest_date)
    else:
        date_max = pd.Timestamp.today()

    if n_days_back:
        date_min = date_max - pd.Timedelta(days=n_days_back)
        date_min = date_min.strftime("%Y-%m-%d")
    else:
        date_min = n_days_back  # none

    return date_min, date_max.strftime("%Y-%m-%d")


### BLOB TRANSFORMATIONS ###


def transform_blob(peh):
    """
    Transform a mym blob (saved with matlab datajoint) to a list of dictionaries like structure.
    """

    # Transform each element of the array into dictionary
    a = list()
    for i in peh:
        a.append(_blob_to_dict(i))

    if len(a) == 1:
        a = a[0]

    return a


def blob_dict_to_df(blob_dict):
    """
    Expand dictionary from transform_blob function to create a trial by trial DataFrame
    """

    # Check if all sizes of the dictionary arrays are the same size
    sizes = list()
    sizes_dict = dict()
    for i in blob_dict.keys():
        sizes.append(len(blob_dict[i]))
        sizes_dict[i] = len(blob_dict[i])

    if len(set(sizes)) == 1:
        # If so, let's create a dictionary for each index of the arrays and append that to a list
        f_size = sizes[0]
        all_trials_list = list()
        for i in range(f_size):
            this_trial_dict = dict()
            this_trial_dict = {key: value[i] for (key, value) in blob_dict.items()}
            all_trials_list.append(this_trial_dict)

        df_protocol_data2 = pd.DataFrame(all_trials_list)
        return df_protocol_data2
    else:
        # If not the same sizes, we cannot convert return empty DF
        print("Not all variables are the same length. Cannot create proper DataFrame")
        result = "\n".join(f"{key}: {value}" for key, value in sizes_dict.items())
        print(result)
        return pd.DataFrame()


def blob_peh_to_df(blob_peh, append_original_columnname=False):
    """
    Expand peh dictionary columns so each event has it's own column
    """

    df_peh = pd.DataFrame(blob_peh)
    dh_peh2 = df_peh.copy()
    # For each column of the dataframe
    for i in df_peh.columns:
        # Expand dictionary columns
        this_column_df = dh_peh2[i].apply(pd.Series)
        # Add original column name to each of the new columns created
        if append_original_columnname:
            this_column_df = this_column_df.add_prefix(i + "_")
        # Replace original column
        dh_peh2 = pd.concat([dh_peh2.drop([i], axis=1), this_column_df], axis=1)

    return dh_peh2


def _blob_to_dict(array_test, parent_fields=None):
    """
    "Private function"
    Recursive transformation of numpy array (saved with matlab datjoint) to dictionary.
    """

    # Set array as writable for further use
    if isinstance(array_test, np.ndarray):
        array_test = array_test.copy()
        array_test.setflags(write=1)

    # Get fieldnames of structure (or "inherit" fieldnames from "parent")
    if parent_fields is None:
        fields_trial = array_test.dtype.names
    else:
        fields_trial = parent_fields

    # Go deeper into the array
    while 1:
        # Get "child" fieldnames
        new_level = array_test[0]
        new_level_fields_trial = new_level.dtype.names

        # Check if fieldnames has changed
        if new_level_fields_trial != fields_trial:
            break

        # Next level deep
        array_test = array_test[0]

    # If "child" level has different fieldnames
    if new_level_fields_trial is not None:
        # If it's only length one, go deeper into the structure and repeat
        if len(array_test) == 1:
            out_array = _blob_to_dict(array_test[0], parent_fields=fields_trial)
        # Transform each of the elements of the child to dictionary recursively
        else:
            a = list()
            for i in array_test:
                a.append(_blob_to_dict(i, parent_fields=fields_trial))

            int_dict = dict()
            for idx, field in enumerate(fields_trial):
                int_dict[field] = a[idx]
                out_array = int_dict

    # If we don't have more fieldnames, presumably we are in latest level
    else:
        out_array = _mymblob_to_dict2(array_test)

    return out_array


def _mymblob_to_dict2(np_array, as_int=True):
    """
    "Private function"
    Last level numpy numpy array transformation to a dictionary.
    (If a field contains a dj.blob.MatStruct array, it transforms it recursively with _blob_to_dict)
    """

    # Last level of recursion, fieldnames on dtype
    fields = np_array.dtype.names

    # Associate each element of array with their fieldname
    out_dict = dict()
    for idx, field in enumerate(np_array):
        # Set array as writable for further use
        if isinstance(field, np.ndarray):
            field = field.copy()
            field.setflags(write=1)
        # If an element is dj.blob.MatStruct, it should be unpacked recursively again
        if isinstance(field, dj.blob.MatStruct):
            out_dict[fields[idx]] = _blob_to_dict(field)
        # If element is array with 1 element, unpack it.
        else:
            l = len(field) if field.shape else 0
            if l == 1:
                field = field[0]

            # Check if variable is indeed a nested structure or dictionary
            this_field_names = field.dtype.names
            # If not just append
            if this_field_names is None:
                out_dict[fields[idx]] = field
            # If it is call blob to dict again
            else:
                a = list()
                for i in field:
                    a.append(_blob_to_dict(i))
                if len(a) == 1:
                    a = a[0]
                out_dict[fields[idx]] = a

    return out_dict


########################
###   SessAggDate    ###
########################


def aggregate_todays_data(animal_ids: list = None) -> pd.DataFrame:
    """
    Utility function used to aggregate today's date for a list of animal_ids
    from the Sessions, Mass, Water, and Rigwater tables in the DataJoint database.
    With an identical structure to the SessAggDate table.

    This is necessary because the SessAggDate table is not updated in real-time
    and is only updated once a day. This function allows us to get the most up-to-date
    information for today's date.

    """

    sessions_df = aggregate_todays_sessions(animal_ids)
    mass_df = aggregate_todays_mass(animal_ids)
    water_df = aggregate_todays_water(animal_ids)
    rigwater_df = aggregate_todays_rigwater(animal_ids)

    todays_df = merge_tables(sessions_df, mass_df, water_df, rigwater_df)

    return todays_df


def merge_tables(df_sessions, df_mass, df_water, df_rigwater):
    """
    Given the dataframes from the aggregate_todays_sessions, aggregate_todays_mass,
    aggregate_todays_water, and aggregate_todays_rigwater functions, this function
    merges them together to create a single dataframe with all the information for
    today's date.
    """

    df_session_mass = pd.merge(df_sessions, df_mass, how="left", on="ratname")
    df_session_mass = df_session_mass.drop(columns="date")

    df_session_mass_water = pd.merge(
        df_session_mass, df_water, how="left", left_on="ratname", right_on="rat"
    )
    df_session_mass_water = df_session_mass_water.drop(columns="date")
    df_session_mass_water["num_water"] = df_session_mass_water["num_water"].fillna(0)

    df_session_mass_water_rigwater = pd.merge(
        df_session_mass_water, df_rigwater, how="left", on="ratname"
    )
    df_session_mass_water_rigwater = df_session_mass_water_rigwater.drop(
        columns="dateval"
    )
    df_session_mass_water_rigwater = df_session_mass_water_rigwater.drop(columns="rat")
    df_session_mass_water_rigwater["num_rigwater"] = df_session_mass_water_rigwater[
        "num_rigwater"
    ].fillna(0)

    return df_session_mass_water_rigwater


def aggregate_todays_sessions(animal_ids: list = None) -> pd.DataFrame:

    # Create Query Keys
    todays_date = datetime.datetime.today().strftime("%Y-%m-%d")
    todays_session_query = {"sessiondate": todays_date}

    if animal_ids is not None:
        todays_session_query = [
            {"ratname": x, "sessiondate": todays_date} for x in animal_ids
        ]

    columns_query = [
        "ratname",
        "sessiondate",
        "n_done_trials",
        "starttime",
        "endtime",
        "hostname",
        "total_correct",
        "percent_violations",
        "right_correct",
        "left_correct",
        "foodpuck",
    ]

    # Query the data
    todays_sessions = pd.DataFrame(
        (bdata.Sessions & todays_session_query).fetch(*columns_query, as_dict=True)
    )

    # If there are no sessions for today, return an empty dataframe, otherwise process the data
    if todays_sessions.shape[0] > 0:
        todays_sessions["sessiondate"] = pd.to_datetime(
            todays_sessions["sessiondate"], format="%Y-%m-%d"
        )
        todays_sessions["total_correct_n"] = (
            todays_sessions["total_correct"] * todays_sessions["n_done_trials"]
        )
        todays_sessions["percent_violations_n"] = (
            todays_sessions["percent_violations"] * todays_sessions["n_done_trials"]
        )
        todays_sessions["right_correct_n"] = (
            todays_sessions["right_correct"] * todays_sessions["n_done_trials"]
        )
        todays_sessions["left_correct_n"] = (
            todays_sessions["left_correct"] * todays_sessions["n_done_trials"]
        )

        # There can be multiple entries for the same date- now we aggregate them
        # to account for that
        todays_sessions_agg1 = todays_sessions.groupby("ratname").agg(
            {
                "sessiondate": [("sessiondate", "min")],
                "n_done_trials": [("num_sessions", "count"), ("n_done_trials", "sum")],
                "hostname": [("hostname", "min")],
                "starttime": [("starttime", "min")],
                "endtime": [("endtime", "max")],
                "total_correct_n": [("total_correct", "sum")],
                "percent_violations_n": [("percent_violations", "sum")],
                "right_correct_n": [("right_correct", "sum")],
                "left_correct_n": [("left_correct", "sum")],
                "foodpuck": [("foodpuck", "mean")],
            }
        )

        todays_sessions_agg1.columns = todays_sessions_agg1.columns.droplevel()

        todays_sessions_agg1["total_correct"] = (
            todays_sessions_agg1["total_correct"]
            / todays_sessions_agg1["n_done_trials"]
        )
        todays_sessions_agg1["percent_violations"] = (
            todays_sessions_agg1["percent_violations"]
            / todays_sessions_agg1["n_done_trials"]
        )
        todays_sessions_agg1["right_correct"] = (
            todays_sessions_agg1["right_correct"]
            / todays_sessions_agg1["n_done_trials"]
        )
        todays_sessions_agg1["left_correct"] = (
            todays_sessions_agg1["left_correct"] / todays_sessions_agg1["n_done_trials"]
        )

        todays_sessions_agg1 = todays_sessions_agg1.reset_index()
    else:
        todays_sessions_agg1 = pd.DataFrame(columns=columns_query)

    return todays_sessions_agg1


def aggregate_todays_mass(animal_ids: list = None) -> pd.DataFrame:

    # Create Query Keys
    todays_date = datetime.datetime.today().strftime("%Y-%m-%d")
    todays_session_query = {"date": todays_date}

    columns_query = ["ratname", "date", "mass", "tech"]

    if animal_ids is not None:
        todays_session_query = [{"ratname": x, "date": todays_date} for x in animal_ids]

    # Query the data
    todays_mass = pd.DataFrame(
        (ratinfo.Mass & todays_session_query).fetch(*columns_query, as_dict=True)
    )

    # There is only every 1 mass entry per day, so we can just return the dataframe
    if todays_mass.shape[0] > 0:
        todays_mass["date"] = pd.to_datetime(todays_date, format="%Y-%m-%d")
    else:
        todays_mass = pd.DataFrame(columns=columns_query)

    return todays_mass


def aggregate_todays_water(animal_ids: list = None) -> pd.DataFrame:

    # Query Keys
    todays_date = datetime.datetime.today().strftime("%Y-%m-%d")
    todays_session_query = {"date": todays_date}

    columns_query = ["rat", "date", "percent_target", "volume"]

    if animal_ids is not None:
        todays_session_query = [{"rat": x, "date": todays_date} for x in animal_ids]

    # Query the data
    todays_water = pd.DataFrame(
        (ratinfo.Water & todays_session_query).fetch(*columns_query, as_dict=True)
    )

    # There can be multiple entries for the same date- now we aggregate them
    if todays_water.shape[0] > 0:

        todays_water["date"] = pd.to_datetime(todays_water["date"], format="%Y-%m-%d")
        todays_water_agg1 = todays_water.groupby("rat").agg(
            {
                "date": [("date", "min")],
                "percent_target": [("percent_target", "max")],
                "volume": [("volume", "max"), ("num_water", "count")],
            }
        )

        todays_water_agg1.columns = todays_water_agg1.columns.droplevel()
        todays_water_agg1 = todays_water_agg1.reset_index()
    else:
        todays_water_agg1 = pd.DataFrame(columns=columns_query + ["num_water"])

    return todays_water_agg1


def aggregate_todays_rigwater(animal_ids: list = None) -> pd.DataFrame:

    # Query Keys
    todays_date = datetime.datetime.today().strftime("%Y-%m-%d")
    todays_session_query = {"dateval": todays_date}

    columns_query = ["ratname", "dateval", "totalvol"]

    if animal_ids is not None:
        todays_session_query = [
            {"ratname": x, "dateval": todays_date} for x in animal_ids
        ]

    # Query the data
    todays_rigwater = pd.DataFrame(
        (ratinfo.Rigwater & todays_session_query).fetch(*columns_query, as_dict=True)
    )

    # There can be multiple entries for the same date- now we aggregate them
    if todays_rigwater.shape[0] > 0:

        todays_rigwater["dateval"] = pd.to_datetime(
            todays_rigwater["dateval"], format="%Y-%m-%d"
        )
        todays_rigwater_agg1 = todays_rigwater.groupby("ratname").agg(
            {
                "dateval": [("dateval", "min")],
                "totalvol": [("totalvol", "max"), ("num_rigwater", "count")],
            }
        )

        todays_rigwater_agg1.columns = todays_rigwater_agg1.columns.droplevel()
        todays_rigwater_agg1 = todays_rigwater_agg1.reset_index()
    else:
        todays_rigwater_agg1 = pd.DataFrame(columns=columns_query + ["num_rigwater"])

    return todays_rigwater_agg1
