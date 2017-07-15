import pandas as pd
import numpy as np
from airbnb import feateng
from sklearn.preprocessing import LabelEncoder, label_binarize, MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
import pickle

"""
Must import the following modules for use on cluster:
module load python/intel/2.7.12
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


def main(data_directory, add_sessions, merge_classes, output_file_name, seed):
    """
    Load the airbnb data set. Fit it with a keras MLP model with specified parameters. Save the result.

    Params:

    Returns:

    """
    # Load the data and generate features
    debug = False
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

    # set up the classifier
    clf = LogisticRegressionCV(Cs=[.0001, .001, .01, .1, 1.0, 10.0, 100.0, 1000.0],
                               penalty='l1',
                               cv=5,
                               max_iter=10000,
                               fit_intercept=True,
                               verbose=2,
                               solver='liblinear',
                               random_state=seed)
    #fit the model to training data
    result = clf.fit(X['train'],Y['train'])

    #save resulting model with pickle
    pickle.dump(result, open(output_file_name, 'wb'))

if __name__ == '__main__':

    #choose params
    data_directory = 'data/'
    add_sessions = 'bin'
    merge_classes = ['AU','CA','DE','ES','FR','GB','IT','NL','PT','other']
    output_file_name = 'logregModel_NDF_other_bin.sav'
    seed = 0

    #print
    print "data_directory: %s" % (data_directory)
    print "add_sessions: %s" % (add_sessions)
    print "merge_classes: %s" % (merge_classes)
    print "output_file_name: %s" % (output_file_name)
    print "seed: %s" % (seed)

    #execute main function
    main(data_directory, add_sessions, merge_classes, output_file_name, seed)