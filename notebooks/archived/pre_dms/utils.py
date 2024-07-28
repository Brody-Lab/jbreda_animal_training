import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj
dj.config["enable_python_native_blobs"] = True
import os
import seaborn as sns
import pandas as pd

sns.set_context("talk")


# def generate_gnp_stage_plot(animal_ids,fig_save_path=None):
#     """
#     Function to visualize fixation growth for an animal given for initial gnp
#     block. Here, this is hard coded to be all the sessions prior to an animal
#     reaching 'always'.

#     inputs
#     ------
#     animal_ids    : list
#         list of animals to generate plot for
#     fig_save_path : str, optional
#         path specifying where to save plots

    
#     returns
#     -------
#     dict w filtered df given gnp date threshold
    
#     plots
#     -----
#     delta vs. session date for sessions that make gnp date threshold

    
#     """

#     # connect to sessions table
#     acquisition = dj.create_virtual_module('new_acquisition', 'bl_new_acquisition')
    
#     gnp_info = {}

#     for animal in animal_ids:
    
#         # make df
#         animal_session_key = {'session_rat' : animal}
#         sessions_df = pd.DataFrame((acquisition.Sessions & animal_session_key).fetch(as_dict=True))

#         # parse comments
#         sessions_df = parse_comments(sessions_df)
        
#         # grab only before always
#         idxs = sessions_df.index[sessions_df['stage_info'] == 'always']
#         if idxs.shape == (0,):
#             continue
        
#         # filter for gnp only given this threshold
#         filt_df = sessions_df.loc[0:idxs[0]]
#         gnp_df = filt_df[filt_df['stage_info'] == 'grow nose poke']
        
#         # calulate days in stage
#         gnp_days = (gnp_df['session_date'].iloc[-1] - gnp_df['session_date'].iloc[0]).days
        
#         # append to dict
#         gnp_info[animal] = {'df': gnp_df, 'ndays': gnp_days}
#         # gnp_info[animal] = dict(['ndays', gnp_days])
            

#         # plot
#         fig, ax = plt.subplots(1,1, figsize=(15, 8))
#         plot_delta_timeline(filt_df, animal, ax=ax, gnp_days=gnp_days)

#         # save
#         fig_name = f"{animal}_gnp_stage_plot"

#         if fig_save_path is None:
#             fig_save_path = os.path.join(os.getcwd(), 'figures')
#         else:
#             fig_save_path = fig_save_path

#         plt.savefig(os.path.join(fig_save_path, fig_name), bbox_inches='tight')
#         plt.close("all")
        
#     return gnp_info

def generate_delta_timline_plot(animal_ids,fig_save_path=None):
    """
    Function to visualize fixation growth (delta) over all sessions for an animal

    inputs
    ------
    animal_ids    : list
        list of animals to generate plot for
    fig_save_path : str, optional
        path specifying where to save plots

    plots
    -----
    delta values by session data conditioned on training stage
    for a list of anaimals
    """

    # connect to sessions table
    acquisition = dj.create_virtual_module('new_acquisition', 'bl_new_acquisition')

    for animal in animal_ids:
        print(animal)
        # make df
        animal_session_key = {'session_rat' : animal}
        sessions_df = pd.DataFrame((acquisition.Sessions & animal_session_key).fetch(as_dict=True))

        # parse comments
        sessions_df = parse_comments(sessions_df)

        # plot
        fig, ax = plt.subplots(1,1, figsize=(15, 8))
        plot_delta_timeline(sessions_df, animal, ax=ax)

        # save
        fig_name = f"{animal}_training_timeline"

        if fig_save_path is None:
            fig_save_path = os.path.join(os.getcwd(), 'figures')
        else:
            fig_save_path = fig_save_path

        plt.savefig(os.path.join(fig_save_path, fig_name), bbox_inches='tight')
        plt.close("all")

def parse_comments(df):
    """
    function that will parse the session_comments string for stage info
    and nose growth duration (delta)

    inputs
    ------
    df : pandas df
        data frame with session info

    returns
    -------
    df : pandas df
        data gframe with appended columns for stage_info and delta values

    """

    # define
    comments = df['session_comments']

    stage_names = ['learning sides', 'learning reward sounds',
                   'grow nose poke', 'always', 'delayed', 'never']

    # init space
    delta = np.zeros(len(comments))
    delta[:] = np.NaN # don't want to plot zeros if empty

    stage_info = np.zeros(len(comments), dtype=object)
    stage_info[:] = np.NaN


    for n_sess, value in comments.items():

        delta_idx = value.find('delta')

        # if you found delta in the comment, grab the numerical value
        # associated with it
        if delta_idx != -1:
            delta[n_sess] = value[delta_idx+6 : delta_idx+9]

        # if one of these stage names is in the comments, grab it
        for stage in stage_names:
            if value.find(stage) != -1:
                stage_info[n_sess] = stage


    # add to the dataframe
    df['delta'] = delta
    df['stage_info'] = stage_info

    return df

def plot_delta_timeline(df, animal,ax=None, gnp_days=None):

    """
    function to plot delta timeline from trianing data

    inputs
    ------
    df     : pandas data frame
        data frame crated by parse_comments()
    animal : str
        name of animal who's data is being plotted
    ax     : str or None, defualt=None
        axis to plot to

    plots
    -------
    delta values by session data conditioned on training stage for an anaimal

    """

    ax = plt.gca() if ax is None else ax

    ax = sns.scatterplot(data=df, x="session_date", y="delta", hue="stage_info")

    # aesthetics
    _ = plt.xticks(rotation=45)
    if gnp_days:
        ax.set(title=f"{animal} days in GNP: {gnp_days}")
    else:
        ax.set(title=f"Training history for {animal}")
    sns.despine()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)
