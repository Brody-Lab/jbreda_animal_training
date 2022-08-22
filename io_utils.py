from multiprocessing.spawn import prepare
import numpy as np
import datajoint as dj
dj.blob.use_32bit_dims = True
import pandas as pd
import pandas as pd
from pathlib import Path
from datetime import date 

## VARIABLES

MAP_SA_TO_SB = {
    12000:3000,
    3000:12000,
}

PROTOCOL_DATA_PATH = 'X:\\jbreda\\animal_data\\protocol_data'

ANIMAL_IDS = ['R500', 'R501', 'R502', 'R503', 'R600']

## FUNCTIONS
#TODO: write a add_history_info function to add things like choice, prev_choice, etc

def fetch_latest_protocol_data(animal_ids=None, save_dir=None,crashed_trials_report=False):
    """
    Function to query bdata via datajoint to get trial by trial
    protocol data for a(n) animal(s), clean it, save out and return
    as a single pandas data frame.

    inputs 
    ------
    animal_ids : list, optional
        animal(s) to query database with, (default = ANIMAL_IDS)
    save_dir : str, optional
        path to directory where data frame will be saved, 
        (default = PROTOCOL_DATA_PATH)

    returns
    -------
    all_animals_protocol_df : data frame
        data frame containing protocol data for every session for
        every animal animal_ids 
    """
    save_dir = PROTOCOL_DATA_PATH if save_dir is None else save_dir
    animal_ids = ANIMAL_IDS if animal_ids is None else animal_ids
    assert type(animal_ids) == list, "animal ids must be in a list"

    animals_protocol_dfs = [] 

    # currently using `bdatatest` table but this may change & 
    # will need to f/u with alvaro luna on which table has
    # protocol data
    bdata = dj.create_virtual_module('new_acquisition', 'bdatatest')
    
    # fetch data, clean it & convert to df for each animal
    for animal_id in animal_ids:
        subject_session_key = {'ratname': animal_id} 

        # protocol data needs to be fetched on it's own from sessions table
        protocol_blobs  = (bdata.Sessions & subject_session_key).fetch('protocol_data', as_dict=True)
        sess_ids, dates = (bdata.Sessions & subject_session_key).fetch('sessid','sessiondate')

        protocol_dicts = convert_to_dict(protocol_blobs)
        protocol_df = make_protocol_df(protocol_dicts, animal_id, sess_ids, dates, crashed_trials_report)
        animals_protocol_dfs.append(protocol_df)
        # using dates because can have multiple sess_ids in one session
        print(f"fetched {len(dates)} sessions for {animal_id}") 
    
    # concatenate across animals & save out 
    all_animals_protocol_df = pd.concat(animals_protocol_dfs, ignore_index=True) 
    date_today = date.today().strftime("%y%m%d")# reformat to YYMMDD 
    file_name = f"{date_today}_protocol_data.csv"
    all_animals_protocol_df.to_csv(Path(save_dir, file_name))

    return all_animals_protocol_df

def convert_to_dict(protocol_blobs):
    """
    Function that takes protocol_data blobs from bdata sessions 
    table query and converts them to python dictionaries

    inputs
    ------
    protocol_blobs : list of arrays
        list returned when fetching protocol_data from Sessions table
        with len = n sessions
    
    returns
    -------
    protocol_dicts : list of dictionaries
        list of protocol_data dictionaries where len = n sessions and 
        each session has a dictionary  
    """
    protocol_dicts = []
    for session in protocol_blobs:
        dict = mymblob_to_dict(session['protocol_data'])
        protocol_dicts.append(dict)
    return protocol_dicts

# TODO- update with new functions written by alvaro for recursive readins
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
        dictionary of protocol_data for a single session with
        keys = to fields of the protocol_data struct
    
    Written by Alvaro
    """
    assert dj.blob.use_32bit_dims, "need to turn on 32 bit dims to read brodylab blobs!"

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

def make_protocol_df(protocol_dicts, animal_id, session_ids, dates, crashed_trials_report):
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
        session ids fetched from sessions table that correspond to 
        values in protocol_dicts
    dates : arr
        dates fetched from sessions table that correspond to 
        values in protocol_dicts 
    
    !note!
        pd data structure must be saved to sessions table with all 
        features having same length (n_started_trials). see DMS or PWM2 
        protocols' HistorySection.m for example in `make_and_send_summary`
        Early stages of testing these new protocols didn't follow this 
        rule & caused bugs that are outlined & fixed via `truncate_sb_length` 
        and `fill_results_post_crash`

    returns:
    -------
    all_sessions_protocol_df: date frame
        protocol_data for every session for an animal 
    """
    session_protocol_dfs = [] 

    # for each session, turn protocol data dict into data frame 
    # and then concatenate together into single data frame
    for isession in range(len(protocol_dicts)):
        prepare_dict_for_df(protocol_dicts[isession]) # ensures correct lengths
        protocol_df = pd.DataFrame.from_dict(protocol_dicts[isession])
        if len(protocol_df)==0: # skip sessions with no trials
            continue    
        clean_protocol_df(protocol_df, animal_id, session_ids[isession], dates[isession], crashed_trials_report)
        session_protocol_dfs.append(protocol_df)
    
    all_sessions_protocol_df = pd.concat(session_protocol_dfs) 
    return all_sessions_protocol_df

def prepare_dict_for_df(protocol_dict):
    """
    Function to clean up a session's protocol dictionary lengths, 
    names & types to ensure there are no errors & interpretation 
    issues upon converting it into a data frame. 

    inputs
    ------
    protocol_dict : dict
        dictionary for a single session's protocol_data

    modifies
    --------
    protocol_dict : dict
        corrects side vector format, updates DMS match/nonmatch
        variable names, corrects for bugs found in HistorySection.m
        that led to differing variables lengths
    """
    # lllrllr to [l, l, l, r....]
    protocol_dict['sides'] = list(protocol_dict['sides'])
    
    # if DMS, convert match/nonmatch category variable to bool
    # with more informative name
    if 'dms_type' in protocol_dict:       
        protocol_dict['is_match'] = protocol_dict.pop('dms_type') 
        protocol_dict['is_match'] = protocol_dict['is_match'].astype(bool)

    # check to see if protocol_data is pre HistorySection.m bug fixes 
    # where len(each value) was not equal. Using sa as reference length
    # template but this could lead to errors if sa has bug
    if len(protocol_dict['sa']) != len(protocol_dict['sb']):
            truncate_sb_length(protocol_dict)
    if len(protocol_dict['sa']) != len(protocol_dict['result']):
            fill_result_post_crash(protocol_dict)
    
    # catch any remaining length errors 
    lens = map(len, protocol_dict.values())
    n_unique_lens = len(set(lens))
    assert n_unique_lens == 1, ("length of dict values unequal!")



def truncate_sb_length(protocol_dict):
    """
    Function to correct for bug in HistorySection, see commit:
    https://github.com/Brody-Lab/Protocols/commit/4a2fadb802d64b7ed66891a263a366a8d2580483
    sb vector was 1 greater than n_started_trials due to an
    appending error

    inputs
    ------
    protocol_dict : dict
        dictionary for a single session's protocol_data
    
    modifies
    -------
    protocol_dict : dict
        updated protocol_dict with sb column length & contents corrected
        if DMS. length only corrected if PWM2 protocol is being used
    """

    # rename for ease
    sa = protocol_dict['sa']
    sb = protocol_dict['sb']
    match = protocol_dict['is_match']

    # if DMS task, can infer values
    if 'is_match' in protocol_dict:
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

def fill_result_post_crash(protocol_dict):
    """
    Function to correct for bug in HistorySection, see commit:
    https://github.com/Brody-Lab/Protocols/commit/3bdde4377ffde011cc34d098acfeb77b74c9e606
    result vector was shorter than n_started_trials because program
    crashed & results_history vector was not being properly filled
    during crash clean up

    inputs
    ------
    protocol_dict : dict
        dictionary for a single session's protocol_data
    
    modifies
    -------
    protocol_dict : dict
        updated protocol_dict with results column length corrected to 
        reflect crash trials
    """

    # rename for ease
    results = protocol_dict['result']
    sa = protocol_dict['sa']

    # pack with crash result value (5)
    crash_appended_results = np.ones((len(sa))) * 5 
    crash_appended_results[0 : len(results)] = results

    protocol_dict['result'] = crash_appended_results



def clean_protocol_df(protocol_df, animal_id, session_id, date, 
                      crashed_trials_report=False):
    """
    Function that takes a protocol_df and cleans it to correct for
    data types and format per JRBs preferences

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
    
    n_started_trials = len(protocol_df)
    # create trials column incase df index gets reset
    protocol_df.insert(0, 'trial', np.arange(1, n_started_trials + 1, dtype=int)) 
    
    # drop any trials where dispatcher reported a crash
    protocol_df.drop(protocol_df[protocol_df['result'] == 5].index, inplace=True) 
    n_completed_trials = len(protocol_df) 
    if crashed_trials_report:
        if n_started_trials > n_completed_trials:
            print(f"{n_started_trials - n_completed_trials} crash trials found for {animal_id} on {session_id}")

    # add animal id, data and sessid value for each trial    
    protocol_df.insert(1, 'animal_id', [animal_id] * len(protocol_df))
    protocol_df.insert(2, 'date', [date] * len(protocol_df)) 
    protocol_df.insert(3, 'session_id', [session_id] * len(protocol_df))

    # convert units to kHz
    protocol_df[['sa', 'sb']] = protocol_df[['sa', 'sb']].apply(lambda row: row / 1000)
    # create a unique pair column & violation column used for sorting in plots
    protocol_df['sound_pair'] = protocol_df.apply(lambda row: str(row.sa) + ', ' + str(row.sb), axis=1)
    protocol_df['violations'] = protocol_df.apply(lambda row: 1 if row.result == 3 else 0, axis=1)
    protocol_df.insert(5, 'violations', protocol_df.pop('violations'))

    # convert data types (matlab makes everything a float)
    int_columns = ['hits', 'violations', 'temperror', 'result', 'helper', 'stage', 'session_id']
    protocol_df[int_columns] = protocol_df[int_columns].astype('Int64')
    category_columns = ['result', 'stage', 'session_id', 'sound_pair']
    protocol_df[category_columns] = protocol_df[category_columns].astype('category')
