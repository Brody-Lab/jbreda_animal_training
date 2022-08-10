import numpy as np
import datajoint as dj
dj.blob.use_32bit_dims = True
import pandas as pd
import pandas as pd


MAP_SA_TO_SB = {
    12000:3000,
    3000:12000,
}

def mymblob_to_dict(np_array, as_int=True):
    '''
    Transform a numpy array to dictionary:
    (numpy array are stored when saving Blobs in 
    MATLAB Datajoint, normally a dictionary will be the fit)
    
    Written by alvaro

    Example: 
    -------
    protocol_dict = mymblob_to_dict(session_data[0]['protocol_data'])
    '''

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

def make_protocol_df(protocol_dict, animal_id, session_date, sessid):
    """
    Takes protocol_dict and turns into dataframe 

    inputs:
    -------
    protocol_dict : dict
        dictionary representing pd.data blob saved out by HistorySection
        and created by mymblob_to_dict
        note : pd must be saved with all features having same length, 
        see DMS and PWM2 protocols for example in `make_and_send_summary`
    
    TODO: add additional inputs 

    returns:
    -------
    protocol_df : pd df
        data frame created from dictionary with crashed trials removed
    """

    # clean up data types for dataframe
    protocol_dict['dms_type'] = protocol_dict['dms_type'].astype(bool)
    protocol_dict['sides'] = list(protocol_dict['sides'])

    # check to see if protocol_data is pre HistorySection bug fix
    if len(protocol_dict['sa']) != len(protocol_dict['sb']):
        correct_sb(protocol_dict)

    # make dataframe & clean
    protocol_df = pd.DataFrame.from_dict(protocol_dict)
    protocol_df = clean_protcol_df(protocol_df, animal_id, session_date, sessid)

    return protocol_df



def correct_sb(protocol_dict):

    """
    Function to correct for bug in HistorySection, see commit:
    4a2fadb802d64b7ed66891a263a366a8d2580483 in Brodylab/Protocols/DMS
    

    inputs
    ------
    protocol_dict : dict
        dictionary representing pd.data blob saved out by HistorySection
        and created by mymblob_to_dict
    
    returns
    -------
    protocol_dict : dict
        updated protocol_dict with sb column length & contents corrected
        if DMS. length only corrected if PWM2
    """

    # rename for ease
    sa = protocol_dict['sa']
    sb = protocol_dict['sb']
    match = protocol_dict['dms_type']

    # if DMS task, can infer values
    if 'dms_type' in protocol_dict:
        print('updating sb values to be correct')
        for trial in range(len(sa)):
            if match[trial]:
                sb[trial] = sa[trial] # update sb
            else:
                assert sa[trial] in MAP_SA_TO_SB, "sa not known"
                sb[trial] = MAP_SA_TO_SB[sa[trial]]
            # elif sa[trial] == 12000.0:
            #     sb[trial] = 3000.0
            # elif sa[trial] == 3000.0:
            #     sb[trial] = 12000.0
            # else:
            #     assert False, 'sa/sb match combo unknown!'
    else:
        print('sb values incorrect, only fixing length')
    
    sb = sb[0:-1] # remove extra entry
    protocol_dict['sb'] = sb # update




def clean_protcol_df(protocol_df, animal_id, session_date, sessid):
    """
    TODO write strings
    """
    # make trials, date, session id & animal id columns
    protocol_df.insert(0, 'trial', range(1, len(protocol_df) + 1)) 
    protocol_df = protocol_df[protocol_df.result != 5] # remove crashes
    protocol_df.insert(1, 'animal_id', [animal_id] * len(protocol_df))
    protocol_df.insert(2, 'date', [session_date] * len(protocol_df)) 
    protocol_df.insert(3, 'sessid', [sessid] * len(protocol_df))

    # convert units to kHz
    protocol_df[['sa', 'sb']] = protocol_df[['sa', 'sb']].apply(lambda x: x / 1000) # convert to kHz

    # convert data types (matlab makes everything a float)
    # TODO make pandas see date as a date
    int_columns = ['hits', 'temperror', 'result', 'helper', 'stage']
    protocol_df[int_columns] = protocol_df[int_columns].astype('Int64')
    category_columns = ['result', 'stage']
    protocol_df[category_columns] = protocol_df[category_columns].astype('category')

    return protocol_df



#TODO: write a add_history_info function to add things like choice, prev_choice, etc