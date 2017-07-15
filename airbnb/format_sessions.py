"""
Purpose of this code is to preprocess the data in the sessions.csv file to see if any of the
features may be useful for prediction.

This code is slow! so preprocessing will speed up model testing.


Each combination of action, action_type, and action_detail defines a feature vector for the user
The entries of the vector are either counts for the number of times that combination occurred or
the number of seconds that those combinations occurred.  The vector itself is 179 features.
Dimensionality reduction may help here since so many features.
Note that we are losing temporal information here. N_grams are a possible mehtod to reintroduce
temporal context information but not a strong reason to think that it will improve performance
for predicing bookings.

Possible representations of data for each user_id
- sum number of times an action,action_type,action_detail taken
- sum number of seconds spent on an action,action_type,action_detail
- sum number of any actions taken
- sum number of secs_elapsed for all actions
- device type used by each user (list of all devices)

Consider running some form of dimensionality reduction on the action list matrix
Goal is to replace huge list of action, type, and detail combinations with a reduced number of latent
factors that summarize the variability in the data.
Be sure to plot variance explained to understand whether reduction was effective.

For example, viewing search results and changing trip characteristics may be highly correlated.
Reducing them to a single latent variable may improve model fitting.

Need to evaluate models in which different features are or are not included.
For example, do we gain anything by including secs_elapsed or is it just redundent?
What about summing number of times an action taken versus just flagging it?

On another note, would be good to try a model in which NDF is ignored and then blend with a more
complete model.  How is the unbalanced data affecting the fit?

"""


import numpy as np
import pandas as pd
import os


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




    # #now we need to map the rows of act_assoc_count and act_assoc_secs
    # #to the rows of our training and test data
    #     #check how large of an intersection between the groups
    # id_list = np.sort(pd.unique(sessions['user_id']))
    # df_all_id_list=list(df_all['id'])
    # id_intersect=np.intersect1d(df_all_id_list,id_list)
    # print("Total intersect with sessions = %d/%d (%2.2f%%)" % (len(id_intersect), len(df_all_id_list),100*(float(len(id_intersect))/float(len(df_all_id_list)))))
    #
    # df_train_id_list=list(df_train['id'])
    # id_intersect=np.intersect1d(df_train_id_list,id_list)
    # print("Train intersect with sessions = %d/%d (%2.2f%%)" % (len(id_intersect), len(df_train_id_list),100*(float(len(id_intersect))/float(len(df_train_id_list)))))
    #
    # df_test_id_list=list(df_test['id'])
    # id_intersect=np.intersect1d(df_test_id_list,id_list)
    # print("Test intersect with sessions = %d/%d (%2.2f%%)" % (len(id_intersect), len(df_test_id_list),100*(float(len(id_intersect))/float(len(df_test_id_list)))))

        #join the dataframes
    #df_all.index=df_all['id']   #set the index of df_all to the id so that we can join with act_assoc dataframes
    #df_all=df_all.join(act_assoc_count,how='left')
    #df_all=df_all.join(act_assoc_secs,how='left')
    #df_all=df_all.join(device_indicators,how='left')
