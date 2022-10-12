
import numpy as np
import datajoint as dj
import pandas as pd


def transform_blob(peh):
    '''
    Transform a mym blob (saved with matlab datajoint) to a list of dictionaries like structure.
    '''

    # Transform each element of the array into dictionary
    a = list()
    for i in peh:
        a.append(_blob_to_dict(i))

    if len(a) == 1:
        a = a[0]

    return a

def blob_dict_to_df(blob_dict):
    '''
    Expand dictionary from transform_blob function to create a trial by trial DataFrame
    '''

    #Check if all sizes of the dictionary arrays are the same size
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
            this_trial_dict = {key:value[i] for (key,value) in blob_dict.items()}
            all_trials_list.append(this_trial_dict)

        df_protocol_data2 = pd.DataFrame(all_trials_list)
        return df_protocol_data2
    else:
        #If not the same sizes, we cannot convert return empty DF
        print('Not all variables are the same length. Cannot create proper DataFrame')
        result = '\n'.join(f'{key}: {value}' for key, value in sizes_dict.items())
        print(result)
        return pd.DataFrame()

def blob_peh_to_df(blob_peh, append_original_columnname=False):
    '''
    Expand peh dictionary columns so each event has it's own column
    '''

    df_peh = pd.DataFrame(blob_peh)
    dh_peh2 = df_peh.copy()
    #For each column of the dataframe
    for i in df_peh.columns:
        #Expand dictionary columns
        this_column_df = dh_peh2[i].apply(pd.Series)
        #Add original column name to each of the new columns created
        if append_original_columnname:
            this_column_df = this_column_df.add_prefix(i+'_')
        # Replace original column
        dh_peh2 = pd.concat([dh_peh2.drop([i], axis=1), this_column_df], axis=1)

    return dh_peh2


def _blob_to_dict(array_test, parent_fields=None):
    '''
    "Private function"
    Recursive transformation of numpy array (saved with matlab datjoint) to dictionary.
    '''

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
    '''
    "Private function"
    Last level numpy numpy array transformation to a dictionary. 
    (If a field contains a dj.blob.MatStruct array, it transforms it recursively with _blob_to_dict)
    '''

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
            l=len(field) if field.shape else 0
            if l==1:
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


def mymblob_to_dict(np_array, as_int=True):
    '''
    DEPRECTATED ------------------------------
     Transform a numpy array to dictionary:
    (numpy array are stored when saving Blobs in  MATLAB Datajoint, normally a dictionary will be the fit)
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
