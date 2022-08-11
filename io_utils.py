import numpy as np
import datajoint as dj
dj.blob.use_32bit_dims = True
import pandas as pd
import pandas as pd
from pathlib import Path

## VARIABLES

MAP_SA_TO_SB = {
    12000:3000,
    3000:12000,
}

PROTOCOL_DATA_PATH = 'X:\\jbreda\\animal_data\\protocol_data'

ANIMAL_IDS = ['R500', 'R501', 'R502', 'R503', 'R600']

## FUNCTIONS
#TODO: write a add_history_info function to add things like choice, prev_choice, etc

def fetch_latest_protocol_data(animal_ids=None, as_dict=True, save_dir=None):
    """
    Function to query bdata via datajoint to get trial by trial
    protocol data for an animal(s), clean it, save out and return.
    Each animal will have all it's sessions collapsed into a single 
    data frame.

    inputs 
    ------
    animal_ids : list, optional
        animal(s) to query database with, (default = ANIMAL_IDS)
    as_dict : bool, optional
        if protocol data frames should be saved into a dictionary 
        with animal_id as key or a list (default = True)
    save_dir : str, optional
        path to directory where data frames will be saved for each
        animal, (default = PROTOCOL_DATA_PATH)

    returns
    -------
    dfs : dict or list
        protocol_data for every session completed by an animal converted
        to a single data frame and stored either as values in a dictionary 
        with animal_ids as keys or a list with len = n animal_ids
    """
    save_dir = PROTOCOL_DATA_PATH if save_dir is None else save_dir
    animal_ids = ANIMAL_IDS if animal_ids is None else animal_ids
    assert type(animal_ids) == list, "animal ids must be in a list"

    dfs = [] # where data frame for each animal will be appended

    # connect to data joint & grab sessions table
    bdata = dj.create_virtual_module('new_acquisition', 'bdatatest')
    
    for animal_id in animal_ids:
        # fetch, convert & clean
        subject_session_key = {'ratname': animal_id} # fetch by animal id
        sess_ids, dates = (bdata.Sessions & subject_session_key).fetch('sessid','sessiondate')
        protocol_blobs  = (bdata.Sessions & subject_session_key).fetch('protocol_data', as_dict=True)
        protocol_dicts = convert_to_dict(protocol_blobs)
        flattened_protocol_df = make_protocol_df(protocol_dicts, animal_id, sess_ids, dates)
        
        # update & save out
        print(f"fetched {len(dates)} sessions for {animal_id}")
        file_name = f"{animal_id}_protocol_data.csv"
        flattened_protocol_df.to_csv(Path(save_dir, file_name))
        dfs.append(flattened_protocol_df)

    if as_dict:
        dfs = dict(zip(animal_ids, dfs))
    
    return dfs

def convert_to_dict(protocol_blobs):
    """
    Function that takes protocol_data blobs from bdata sessions 
    table query and converts them to python dictionaries

    inputs
    ------
    protocol_blobs : list of
        list returned when fetching protocol_data from Sessions table
        with len = n sessions
    
    returns
    -------
    protocol_dicts : list of ditonaries
        list of protocol_data dictionaries where len = n sessions and 
        each session has a dictionary  
    """
    protocol_dicts = []
    for session in protocol_blobs:
        dict = mymblob_to_dict(session['protocol_data'])
        protocol_dicts.append(dict)
    return protocol_dicts

def mymblob_to_dict(np_array, as_int=True):
    """
    Transform a numpy array to dictionary:
    (numpy array are stored when saving Blobs in 
    MATLAB Datajoint, normally a dictionary will be the fit)

    inputs:
    -------
    np_array : arr
        array of protocol_data for a single session
    
    returns:
    --------
    out_dict : dict
        dictonary of protocol_data for a single session with
        keys = to fields of the protocol_data struct
    
    Written by Alvaro
    """

    # Transform numpy array to DF
    out_dict = pd.DataFrame(np_array.flatten())

    # Flatten each column and get the "real value of it"
    out_dict = out_dict.applymap(lambda x: x.flatten())

    #Get not empty columns to extract fist value of only those columns
    s = out_dict.applymap(lambda x: x.shape[0])
    not_empty_columns = s.loc[:, ~(s == 0).any()].columns.to_list()
    out_dict = out_dict.apply(lambda x: x[0].flatten() if x.name in not_empty_columns else x)
    
    if not isinstance(out_dict, pd.DataFrame):
        out_dict = out_dict.to_frame()
        #Get columns that are "real" arrays not unique values disguised
        s = out_dict.applymap(lambda x: x.size).T
        real_array_columns = s.loc[:, (s > 1).any()].columns.to_list()
        out_dict = out_dict.T
        out_dict = out_dict.apply(lambda x: x[0] if x.name not in real_array_columns else x, axis=0)
        
    columns = out_dict.columns.copy()
    out_dict = out_dict.squeeze()

    #Transform numeric columns to int (normally for params)
    if as_int:
        for i in columns:
            if (isinstance(out_dict[i],np.float64) and out_dict[i].is_integer()):
                out_dict[i] = out_dict[i].astype('int')

    out_dict = out_dict.to_dict()

    return out_dict

def make_protocol_df(protocol_dicts, animal_id, session_ids, dates):
    """
    Converts 

    inputs:
    -------
    protocol_dicts : list of dictionaries
        protocol_data dictionary for each session queried 
    animal_id : str
        id of animal that protocol_data is associated with, note this
        function currently assumes 1 animal per query
    session_ids : arr
        session ids fetched from sessions table that correspond to a
        protocol_data dictionary in protocol_dicts
    dates : arr
        dates fetched from sessions table that correspond to a
        protocol_data dictionary in protocol_dicts 
    
    note : pd data structure must be saved to sessions table with all 
        features having same length, see DMS and PWM2 protocols for 
        example in `make_and_send_summary`

    returns:
    -------
    flattened_protocol_df: date frame
        protocol_data for each session flattened into a single data frame for
        an animal that has been cleaned 
    """
    protocol_dfs = [] # where data frame for each session with be appended

    for isession in range(len(protocol_dicts)):
        # clean up data types for dataframe
        protocol_dicts[isession]['dms_type'] = protocol_dicts[isession]['dms_type'].astype(bool)
        protocol_dicts[isession]['sides'] = list(protocol_dicts[isession]['sides'])

        # check to see if protocol_data is pre HistorySection bug fix
        if len(protocol_dicts[isession]['sa']) != len(protocol_dicts[isession]['sb']):
            correct_sb(protocol_dicts[isession])

        # make dataframe & clean
        protocol_df = pd.DataFrame.from_dict(protocol_dicts[isession])
        protocol_df = clean_protocol_df(protocol_df, animal_id, session_ids[isession], dates[isession])
        protocol_dfs.append(protocol_df)
    
    flattened_protocol_df = pd.concat(protocol_dfs) # collapse across sessions

    return flattened_protocol_df

def correct_sb(protocol_dict):
    """
    Function to correct for bug in HistorySection, see commit:
    4a2fadb802d64b7ed66891a263a366a8d2580483 in Brodylab/Protocols/DMS

    inputs
    ------
    protocol_dict : dict
        dictionary for a single sessions protocol_data
    
    modifies
    -------
    protocol_dict : dict
        updated protocol_dict with sb column length & contents corrected
        if DMS. length only corrected if PWM2 protocol is being used
    """

    # rename for ease
    sa = protocol_dict['sa']
    sb = protocol_dict['sb']
    match = protocol_dict['dms_type']

    # if DMS task, can infer values
    if 'dms_type' in protocol_dict:
        for trial in range(len(sa)):
            if match[trial]:
                sb[trial] = sa[trial] # update sb
            else:
                assert sa[trial] in MAP_SA_TO_SB, "sa not known"
                sb[trial] = MAP_SA_TO_SB[sa[trial]]
    else:
        print('sb values incorrect, only fixing length')
    
    sb = sb[0:-1] # remove extra entry
    protocol_dict['sb'] = sb # update

def clean_protocol_df(protocol_df, animal_id, session_id, date):
    """
    Function that takes a protocol_df generated from a protocol_data 
    dictionary and cleans it to correct for data types and format
    per JRBs taste

    inputs
    ------
    protocol_df : data frame
        protocol_data dictionary thats been converted to df for a single session
    animal_id : str
        animal id for which the session corresponds to
    session_id : str
        id from bdata corresponding to session
    date : datetime object or str
        date corresponding to session
    
    modifies
    -------
    protocol_df : data frame
        (1) crashed trials removed
        (2) animal, date, session id columns added
        (3) sa/sb converted from Hz to kHz
        (4) certain columns converted to ints & categories
    """
    # make trials, date, session id & animal id columns
    protocol_df.insert(0, 'trial', range(1, len(protocol_df) + 1)) 
    protocol_df = protocol_df[protocol_df.result != 5] # remove crashes
    protocol_df.insert(1, 'animal_id', [animal_id] * len(protocol_df))
    protocol_df.insert(2, 'date', [date] * len(protocol_df)) 
    protocol_df.insert(3, 'sessid', [session_id] * len(protocol_df))

    # convert units to kHz
    protocol_df[['sa', 'sb']] = protocol_df[['sa', 'sb']].apply(lambda x: x / 1000) # convert to kHz

    # convert data types (matlab makes everything a float)
    int_columns = ['hits', 'temperror', 'result', 'helper', 'stage']
    protocol_df[int_columns] = protocol_df[int_columns].astype('Int64')
    category_columns = ['result', 'stage']
    protocol_df[category_columns] = protocol_df[category_columns].astype('category')

    return protocol_df