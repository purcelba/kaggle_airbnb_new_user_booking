import pandas as pd
import numpy as np
from airbnb import feateng
from sklearn.preprocessing import LabelEncoder, label_binarize, MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
import pickle

"""
Fit an L1-regularized logistic regression model implemented in scikit-learn to the kaggle-
airbnb data set using 5-fold cross-validation to find the optimal regularization term.

If running on HPC computing cluster, must import the following modules:
module load python/intel/2.7.12
module load scikit-learn/intel/0.18.1
"""

# set up functions
def split_train_test(df_train, labels):
    """
    Split training data into train and holdout sets (last 10% of data).
    Params:
    - df_train, DataFrame, feature matrix (obs x feats)
    - labels, DataFrame, class labels (obs x 1)
    Returns:
    X, dict, feature matrix for training and holdout data sets are stored in the 'train' 
             and 'holdout' keys, respectively
    Y, dict, class labels for training and holdout data sets in 'train' and 'holdout'
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

def main(data_directory, add_sessions, merge_classes, rescale_predictors, output_file_name, seed):
    """
    Fit L1-regularized logistic regression model to the kaggle-airbnb dataset using 5-fold
    cross-validation to find the regularization parameter.  The fitted model is saved as a
    pickle (sav file) to be loaded later.
    
    Params:
    - data_diretory, str, path to directory in which data are stored
    - add_sessions, str, can be 'none','bin', 'count', or 'secs'
        - 'none' will not add any sessions features
        - 'bin' will add binary features indicating 1 if a a user took an action was taken and 0 otherwise
        - 'count' will add integer features indicating the number of times a user took an action
        - 'secs' will add integer features indicating the number of seconds users spent on each action
    - merge_classes, list of str, class names to be merged into a single class.  For example, to fit
        only booking vs non-booking (NDF), then set merge_classes = ['other','US','FR','CA','GB','ES','IT','PT','NL','DE','AU']
    - rescale_predictors, logical, if True, then will rescale predictors in the feature matrix between 0 and 1.
    - output_file_name, str, name of txt file in which to write results.
    - seed, int, fix the PRNG seed

    Returns:
      No variables returned.  Model results are saved as sav file for reloading.

    """
    # Load the data and generate features
    debug = False
    rm_classes = []
    training_data = data_directory + 'train_users_2.csv'
    test_data = data_directory + 'test_users.csv'
    df_train, df_test, labels, id_train, id_test = feateng.feateng1(training_data, test_data, add_sessions, rm_classes, merge_classes,
                                                                    rescale_predictors, debug)

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

    # set up the classifier
    clf = LogisticRegressionCV(Cs=[.0001,.001, .01, .1, 1.0, 10.0, 100.0, 1000.0],
                               penalty='l1',
                               cv=5,
                               max_iter=10000,
                               fit_intercept=True,
                               verbose=2,
                               solver='liblinear',
                               random_state=seed,
                               n_jobs = 1)
    #fit the model to training data
    result = clf.fit(X['train'],Y['train'])

    #save resulting model with pickle
    pickle.dump(result, open(output_file_name, 'wb'))

if __name__ == '__main__':

    #choose params
    data_directory = 'data/'
    add_sessions = 'none'
    merge_classes = []#['other','US','FR','CA','GB','ES','IT','PT','NL','DE','AU']
    rescale_predictors = True
    output_file_name = 'logregModel_nonesesh.sav'
    seed = 0

    #print
    print "data_directory: %s" % (data_directory)
    print "add_sessions: %s" % (add_sessions)
    print "merge_classes: %s" % (merge_classes)
    print "rescale_predictors: %s" % (rescale_predictors)
    print "output_file_name: %s" % (output_file_name)
    print "seed: %s" % (seed)

    #execute main function
    main(data_directory, add_sessions, merge_classes, rescale_predictors, output_file_name, seed)
    