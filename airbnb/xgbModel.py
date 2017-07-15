import pandas as pd
import numpy as np
from airbnb import feateng
from sklearn.preprocessing import LabelEncoder, label_binarize, MinMaxScaler
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

"""
Must import the following modules for use on cluster:
module load python/intel/2.7.12
module load keras/2.0.2
module load tensorflow/python2.7/20170218
module load h5py/intel/2.7.0rc2
module load scikit-learn/intel/0.18.1
"""


# set up functions
def split_train_test(df_train, labels):
    """
    Split training data into train and holdout sets (last 10% of data).
    """
    n_train = np.shape(df_train)[0]
    X = {'train': [], 'holdout': []}  # features
    Y = {'train': [], 'holdout': []}  # labels
    p10 = int(0.1 * n_train)
    X['holdout'] = df_train.iloc[-p10:]
    Y['holdout'] = labels[-p10:]
    X['train'] = df_train.iloc[:(n_train - p10)]
    Y['train'] = labels[:(n_train - p10)]
    return X, Y


def rescale_predictors(X):
    """
    Rescale predictors between zero and one before fitting.
    Prevents variables with larger variance from dominating prediction error.
    """
    for k in X.keys():
        scaler = MinMaxScaler().fit(X[k])
        X[k] = pd.DataFrame(scaler.transform(X[k]), columns=X[k].columns)
    return X


def main(data_directory, add_sessions, output_file_name, seed, n_iter):
    """
    Params:

    Returns:

    """
    # Load the data and generate features
    debug = False
    merge_classes = []
    rm_classes = []
    training_data = data_directory + 'train_users_2.csv'
    test_data = data_directory + 'test_users.csv'
    df_train, df_test, labels = feateng.feateng1(training_data, test_data, add_sessions, rm_classes, merge_classes,
                                                 debug)

    # print some information about the data
    n_train, n_feats = np.shape(df_train)
    n_test = np.shape(df_test)[0]
    n_labels = np.shape(labels)[0]
    print "%d training observations" % (n_train)
    print "%d test observations" % (n_test)
    print "%d features" % (n_feats)
    print "%d labels" % (n_labels)

    # encode labels as integers
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels).ravel()
    bins_ = np.arange(len(le.classes_))
    bin_labels = le.inverse_transform(bins_)

    # split training data into train and holdout sets (last 10% of data)
    X, Y = split_train_test(df_train, labels)

    # If requested, rescale predictors between zero and one.
    X = rescale_predictors(X)

    # set up randomized search parameter distributions
    import scipy.stats as st
    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)
    params = {
        "n_estimators": st.randint(3, 40),
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        'reg_alpha': from_zero_positive,
        "min_child_weight": from_zero_positive,
    }

    # try xgboost implementation. best current model.
    xgb = XGBClassifier(max_depth=6,
                        learning_rate=0.3,
                        n_estimators=25,
                        objective='multi:softprob',
                        subsample=0.5,
                        colsample_bytree=0.5,
                        seed=seed)
    # execute randomized search with xgboost
    from sklearn.model_selection import RandomizedSearchCV
    gs = RandomizedSearchCV(xgb, params, n_jobs=-1, n_iter=n_iter, verbose=2)

    #save resulting model with pickle
    pickle.dump(gs, open(output_file_name, 'wb'))

if __name__ == '__main__':

    #choose params
    data_directory = 'data/'
    add_sessions = 'bin'
    output_file_name = 'xgbModel_bin.sav'
    seed = 0
    n_iter = 5

    #print
    print "data_directory: %s" % (data_directory)
    print "add_sessions: %s" % (add_sessions)
    print "output_file_name: %s" % (output_file_name)
    print "seed: %s" % (seed)

    #execute main function
    main(data_directory, add_sessions, output_file_name, seed, n_iter)