import numpy as np
import pandas as pd
import os
"""
Preprocess the data in the sessions.csv file.  Encode each combination of action_name, action_type,
and action_detail as a separate variable.  

Warning - this code is slow, but results an be saved and loaded for efficient model testing.

Each combination of action, action_type, and action_detail defines a new feature. 
The entires of the feature vectors can be (1) "bin": binary indicating 1 if the combination occured and zero
otherwise, (2) "counts":  the number of times that combination occurred for a given user, or (3)
the number of seconds that those combinations occurred.  In total 179 features are generated.
"""

# get an array of indicator variables for devices used
# columns = devices, rows = users
def getDeviceIndicators(sessions, **kwargs):
    """
    Given the array of sessions information, return an array of users x device flags.
    Rows are users, columns are indicator variables determin

    Params:
    -sessions, DataFrame, sessions information imported from sessions.csv file.
    Keyword args:
    -verbose, logical default True, if True then display progress.
    Returns:
    - device_indicators, DataFrame, columns are indicator variables determining whether each user used a device.
    """
    # default keyword args
    verbose = kwargs.get("verbose", True)
    # initialize
    device_list = pd.unique(sessions['device_type'])
    id_list = np.sort(pd.unique(sessions['user_id']))
    id_list = id_list[~pd.isnull(id_list)]
    device_flag = np.zeros([len(id_list), len(device_list)])
    id_flag = []
    # reduce the sessions array to accelerate processing
    sessions_reduced = sessions
    sessions_reduced = sessions_reduced.drop(['action', 'action_type', 'action_detail', 'secs_elapsed'], 1)
    # get a pandas groupby object for user_id, begin looping over users
    g = sessions_reduced.groupby('user_id')
    counter = 0
    disp_list = np.arange(0, len(id_list), 500)  # display an update every 500 users
    for id, id_group in g:
        g_device_list = id_group.groupby('device_type')
        for device, device_group in g_device_list:
            index = [x for x, y in enumerate(device_list) if y == device]
            device_flag[counter, index] = 1
            # print(id,device,index)
        id_flag.append(id)
        if counter in disp_list and verbose:
            print("%d/%d finished (%2.2f%%)" % (counter, len(id_list), 100 * (float(counter) / float(len(id_list)))))
            # print(id,action,action_type,action_detail,action_detail_group.shape[0],np.sum(action_detail_group['secs_elapsed']),index)
        counter = counter + 1
    # convert to dataframe
    device_indicators = pd.DataFrame(device_flag, columns=device_list, index=id_flag)
    return device_indicators


# get each possible combination of action, action types, and action details
def getActAssoc(sessions):
    """
    Given the array of sessions data, generate for each user a feature vector where each feature is determined
    by every possible combination of actions, action associations, and action details ("actions").  The entries can either be
    binary (whether the user took the action or not), counts (how many times total did the user take each action),
    or times (how many elapsed seconds total did the user spend on each action).

    Params:
    - sessions, DataFrame, the session information imported from the .csv file.
    Returns
    - act_assoc_count, DataFrame, users x actions, total counts for each action
    - act_assoc_bin, DataFrame, users x actions, 1 or 0 indicating whether the action was taken or not
    - act_assoc_secs, DataFrame, users x actions, total elapsed seconds for each action
    """

    # group by action, action_type, and action_details
    sessions_reduced = sessions  # make a copy
    sessions_reduced = sessions_reduced.drop(['user_id', 'device_type', 'secs_elapsed'],
                                             1)  # reduce to columns of interest
    g = sessions_reduced.groupby(['action', 'action_type', 'action_detail'])  # group by interesting columns
    # save each combination of responses in a list of tuples
    act_assoc = []
    for name, group in g:
        act_assoc.append(name)
    del sessions_reduced

    # for each user, compute the summed times each action, action tpe, and action detail was performed.
    # initialize numpy array to save counts/total sec for each feature
    id_list = np.sort(pd.unique(sessions['user_id']))
    id_list = id_list[~pd.isnull(id_list)]
    act_assoc_count = np.zeros([len(id_list), len(act_assoc)])
    act_assoc_secs = np.zeros([len(id_list), len(act_assoc)])
    id_flag = []
    # note: may also need to save user vector if sorting id_list doesn't match.
    grouped_id = sessions.groupby('user_id')  # group users
    # iterate over users
    counter = 0
    disp_list = np.arange(0, len(id_list), 500)  # display an update every 500 users
    for id, id_group in grouped_id:
        grouped_action = id_group.groupby('action')  # group actions for this user
        for action, action_group in grouped_action:
            grouped_action_type = action_group.groupby('action_type')  # group action types for this action
            for action_type, action_type_group in grouped_action_type:
                grouped_action_detail = action_type_group.groupby(
                    'action_detail')  # group action details for this action type
                for action_detail, action_detail_group in grouped_action_detail:
                    index = [x for x, y in enumerate(act_assoc) if
                             y[0] == action and y[1] == action_type and y[2] == action_detail]
                    act_assoc_count[counter, index] = action_detail_group.shape[0]
                    act_assoc_secs[counter, index] = np.sum(action_detail_group['secs_elapsed'])
        id_flag.append(id)
        counter = counter + 1
        if counter in disp_list:
            print("%d/%d finished (%2.2f%%)" % (counter, len(id_list), 100 * (float(counter) / float(len(id_list)))))
            # print(id,action,action_type,action_detail,action_detail_group.shape[0],np.sum(action_detail_group['secs_elapsed']),index)
    # convert counts to binary
    act_assoc_bin = act_assoc_count.copy()
    act_assoc_bin[act_assoc_bin > 0] = 1
    # Put results in a DataFrame with user indicated
    act_assoc_count = pd.DataFrame(act_assoc_count, columns=act_assoc, index=id_flag)
    act_assoc_secs = pd.DataFrame(act_assoc_secs, columns=act_assoc, index=id_flag)
    act_assoc_bin = pd.DataFrame(act_assoc_bin, columns=act_assoc, index=id_flag)

    # reurn
    return act_assoc_count, act_assoc_bin, act_assoc_secs

def main(sessions_path, save_dir, **kwargs):
    """
    Wrapper script for formatting sessions data and saving as csv.
    
    Params:
    - sessions_path, str, path to the sessions.csv file to be loaded
    - save_dir, str, directory in which the generated csv files will be saved
    Keywords:
    - debug , logical, which or not to drop all but first 1000 observations for fast debugging (default = False)
    
    Returns:
    - No variables returned.  Results are saved as .csv in requested directory.
    
    """
    #set prng seed
    np.random.seed(0)
    #load sessions data
    print "Loading data..."
    sessions = pd.read_csv(sessions_path)         #load training data from csv file to dataframe
    print "Data loaded."
    #convert all -unknown- to nan
    sessions.replace('-unknown-',np.nan,inplace=True)
    # default keyword args
    debug = kwargs.get("debug", False)
    if debug:
        sessions = sessions[:1000]
    #Generate indicator variables for device usage
    print "Generating indicator variables for device usage..."
    device_indicators = getDeviceIndicators(sessions)
    print "Device indicators generated."
    #Generate counts, binaries, and seconds elapsed for each action
    print "Generating indicator variables for actions..."
    act_assoc_count, act_assoc_bin, act_assoc_secs = getActAssoc(sessions)
    print "Action indicator variables generated."
    #save the results as csv
    device_indicators.to_csv(save_dir + 'device_indicators.csv')
    act_assoc_count.to_csv(save_dir + 'act_assoc_count.csv')
    act_assoc_bin.to_csv(save_dir + 'act_assoc_bin.csv')
    act_assoc_secs.to_csv(save_dir + 'act_assoc_secs.csv')
    #indicate completion
    print('Done!')


if __name__ == '__main__':
    #path to sessions.csv file
    sessions_path = '../data/sessions.csv'
    #path to save the dataframes
    save_dir = '../data/'
    #debug
    debug_ = False
    main(sessions_path,save_dir,debug=debug_)
