# -*- coding: utf-8 -*-
"""
airbnbn_feature_engineering.py

Generate an N x P feature matrix, where N are observations (users) and P are features.  

The results will be saved in dataframes df_train (training set), df_all (test_set), and labels (training set labels for predictor variable)

All three dataframes are saved as csv. files in the cache folder using following naming convention:
    df_train_v1.csv
    df_test_v1.csv
    labels.csv
where "v1" will be replaced with "v2, v3, etc ..." depending on the version number.


Several variables contain missing data:
    age (continuous)
    first_affiliate_tracked (categorical)
    gender (categorical)
    language (categorical)

"""

import numpy as np
import pandas as pd
import datetime


def feateng1(training_data, test_data, add_sessions, rm_classes, merge_classes, add_logreg_filename, rescale_predictors, debug):
    """
    Args:
    - training_data, str, path to the training data (.csv file)
    - test_data, str, path to the test data (.csv file)
    - debug, logical, True for debug mode which will eliminate most of the data for fast protoyping

    Returns:
    - df_train, dataframe, formatted feature matrix of training data (observations x features) 
    - df_test, dataframe, formatted feature matrix of test data (obs x feats)
    - labels, dataframe, labels associated with each training observation
    - id_test, dataframe, unique id for each observation in test data. Used to generate submission file.
  
    """
    
    #set debugging mode here.
    if debug:
        print("*************************")
        print("debug mode is on. dropping observations.")
        print("*************************")
    
    #Load data
    print("Loading data...\n")
    df_train = pd.read_csv(training_data)         #load training data from csv file to dataframe
    df_test = pd.read_csv(test_data)             #load testing data csv -> dataframe
    labels = df_train['country_destination']                        #separate labels (i.e., target variable for prediction)
    df_train = df_train.drop(['country_destination'], axis=1)       #remove target variable from training data.
    id_train = df_train['id']
    id_test = df_test['id']
    print("Done")
    
    #save number of training examples to split data later
    n_train = df_train.shape[0]

    #merge classes if requested
    if merge_classes:
        labels[labels.isin(merge_classes)] = 'merged'

    #combine training and test data, clear old variables
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    id_all = pd.concat((id_train, id_test), axis=0, ignore_index=True)
    df_all = df_all.set_index(id_all) #set index
    labels.reindex(id_all)
    del(df_train,df_test)

    #if debugging, keep only 10000 observations for speed
    if debug:
        df_all = pd.concat((df_all[:10000], df_all[-10000:]), axis=0, ignore_index=False)
        labels = pd.concat((labels[:10000], labels[-10000:]), axis=0)
        n_train = 10000-1

    #remove classes if requested
    if rm_classes:
        raise ValueError('Option disabled')

    #convert '-unknown- to NaN to count as missing value in several variables
    df_all.replace('-unknown-',np.nan,inplace=True)

    #remove variables not used for prediction.
    id_all = df_all['id']
    df_all = df_all.drop(['id'], axis=1)                    #id. not useful for prediction.
    df_all = df_all.drop(['date_first_booking'], axis=1)    #date_first_booking.  Absent in test set.

    ####Feature engineering####
    #    1. Format non-categorical variables: date_account_created, timestamp_first_active, age
    #    2. One-hot feature encoding for categorical variables: gender, signup_method, signup_flow, language, affiliate_channel, affiliate_provider, first_affiliate_tracked, signup_app, first_device_type, first_browser
    #    3. Replace all NULL values (e.g., NaNs) as -1 to be modeled as a separate category.
    print("Feature engineering...")
    #date_account_created
    print("    formatting date_account_created...")
        #break up date into year, mo, and day.  Format = YYYY-MM-DD
    dac=np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    df_all['dac_year']=dac[:,0]  #create col for year
    df_all['dac_month']=dac[:,1] #create col for month
    df_all['dac_day']=dac[:,2]   #create col for day
    df_all=df_all.drop(['date_account_created'],axis=1) #remove old col    
        #add day of week (0=mon to 6=sun)
    dac_wkday = [datetime.datetime(df_all['dac_year'][i],df_all['dac_month'][i],df_all['dac_day'][i]).weekday() for i in range(df_all.shape[0])]
    df_all['dac_wkday']=dac_wkday
        #add week of year
    dac_week = [datetime.date(df_all['dac_year'][i],df_all['dac_month'][i],df_all['dac_day'][i]).strftime("%U") for i in range(df_all.shape[0])]
    df_all['dac_week'] = list(map(int,dac_week))
    
    #timestamp_first_active
    print("    formatting timestamp_first_active")
        #break into year, mo, and day.  Format = YYYYMMDDHHMMSS (year, month, day, hour, min second)
    tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
    df_all['tfa_year'] = tfa[:,0] #year is first four numbers, all col to df_all
    df_all['tfa_month'] = tfa[:,1]#month is next two, all col to df_all
    df_all['tfa_day'] = tfa[:,2]  #day is next two, all col to df_all
    df_all = df_all.drop(['timestamp_first_active'], axis=1)#drop old column    
        #add days of week
    tfa_wkday = [datetime.datetime(df_all['tfa_year'][i],df_all['tfa_month'][i],df_all['tfa_day'][i]).weekday() for i in range(df_all.shape[0])]
    df_all['tfa_wkday']=tfa_wkday
        #add week of year
    tfa_week = [datetime.date(df_all['tfa_year'][i],df_all['tfa_month'][i],df_all['tfa_day'][i]).strftime("%U") for i in range(df_all.shape[0])]
    df_all['tfa_week'] = list(map(int,tfa_week))
    
    #age
    print("    formatting age")
        #remove values under 14 and over 100.  note that this is only one way of dealing with problem.
    av = df_all.age.values
    df_all['age'] = np.where(np.logical_or(av<14, av>100), np.nan, av)

    #if requested add the sessions information
    if add_sessions != 'none':
        sessions = pd.read_csv("../data/act_assoc_%s.csv" % (add_sessions),index_col=0)
        sessions['sessions_-1'] = 0                                                         #add a column to flag missing values
        df_all = df_all.join(sessions,how='left')
            # encode missing values in this column from NaN to 1
        df_all['sessions_-1'] = df_all['sessions_-1'].fillna(1)


    #Replace all missing values (nan's) with -1.
    df_all = df_all.fillna(-1)

    #One-hot-encoding features: convert categorical variables into dummy variables.
    #NaNs here are just ignored, but by encoding missing values as -1 we will encode them with an indicator variable here..
    print("    one-hot-feature encoding")
    ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    for f in ohe_feats:
        df_all_dummy = pd.get_dummies(df_all[f], prefix=f) #get dummy variables for this feature.
        df_all = df_all.drop([f], axis=1)                  #drop the old cateogircal column
        df_all = pd.concat((df_all, df_all_dummy), axis=1) #append the dummy variables to df_all

    #One continuous varialbe (age) has missing values.
    #Encode misisng values as zero and set up a separate indicator variable for missing age values
    df_all['age_-1'] = np.zeros(np.shape(df_all['age']))    #create indicator variable for missing age vaues
    df_all.loc[df_all['age'] == -1, 'age_-1'] = 1.0
    df_all.loc[df_all['age'] == -1, 'age'] = 0.0            #set missing values in age column to zero.

    #rescale predictors if requested
    if rescale_predictors:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(df_all)
        df_all = pd.DataFrame(scaler.transform(df_all), columns = df_all.columns)

    #if requested, load a logistic regression model and generate predictions for
    #classification of NDF versus all others.
    if add_logreg_filename:
        #imports
        import pickle
        from sklearn.externals import joblib
        from sklearn.preprocessing import MinMaxScaler
        #load the model
        loaded_model = joblib.load(add_logreg_filename)
        result = loaded_model
        #generate predictions and append
        pred_prob = result.predict_proba(np.array(df_all))
        #convert prob to log-odds
        log_odds = np.log(pred_prob/(1 - pred_prob))
        #rescale if requested
        if rescale_predictors:
            scaler = MinMaxScaler().fit(log_odds)
            log_odds = scaler.transform(log_odds)
        #append to dataframe
        df_all['log_odds_pred'] = log_odds[:,0]

    #Split train/test set
    df_train = df_all[:n_train]
    df_test = df_all[n_train:]

    #convert labels series to dataframe for saving
    labels = labels.to_frame()

    #return
    return df_train, df_test, labels, id_train, id_test
    
#standard boilerplate call
if __name__ == '__main__':
    #options
    training_data = 'data/train_users_2.csv'         #path to training data
    test_data = 'data/test_users.csv'                #path to test data
    add_sessions = 'bin'                                #'none','bin','count','secs'
    rm_classes = []                                     #list of classes to remove, if empty remove none.
    merge_classes = []          #['other','US','FR','CA','GB','ES','IT','PT','NL','DE','AU']
    add_logreg_filename = 'logregModel_binsesh_merged.sav'                            #"../output/logregModel_NDF_other_bin.sav"
    rescale_predictors = True
    debug=False                                          #run in debug mode? (toss observations)

    #run
    feateng1(training_data, test_data, add_sessions, rm_classes, merge_classes, add_logreg_filename, rescale_predictors, debug)