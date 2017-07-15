#To run on HPC
#
#Must import the following modules:
# module load python/intel/2.7.12
# module load keras/2.0.2
# module load tensorflow/python2.7/20170218
# module load h5py/intel/2.7.0rc2 (if saving models)
#
# To do:
# (1) test environmental variable inputs
# (2) test gpu speed improvement
# (3) add some printing in json format for compiling results
# (4) consider also recording binary_crossentropy for each model
#     and also writing custom function to recording likelihood
#
import json
import os
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import feateng
from sklearn.preprocessing import LabelEncoder, label_binarize, MinMaxScaler

def split_train_test(df_train,labels):
    """
    Split training data into train and holdout sets (last 10% of data).
    """
    n_train = np.shape(df_train)[0]
    X = {'train': [],'holdout': []} #features
    Y = {'train':[],'holdout': []} #labels
    p10 = int(0.1*n_train)
    X['holdout'] = df_train.iloc[-p10:]
    Y['holdout'] = labels[-p10:]
    X['train'] = df_train.iloc[:(n_train-p10)]
    Y['train'] = labels[:(n_train-p10)]
    return X,Y

def rescale_predictors(X):
    """
    Rescale predictors between zero and one before fitting.
    Prevents variables with larger variance from dominating prediction error.
    """
    for k in X.keys():
        scaler = MinMaxScaler().fit(X[k])
        X[k] = pd.DataFrame(scaler.transform(X[k]), columns = X[k].columns)
    return X

def main(data_directory, output_filename, seed, n_iter, cv, batch_size, epochs, learn_rate, momentum, init_mode, activation, neurons, layers):

    #Load the data and generate features
    debug = False
    merge_classes = []
    rm_classes = []
    add_sessions = 'none'
    training_data = data_directory + 'train_users_2.csv'
    test_data = data_directory + 'test_users.csv'
    df_train, df_test, labels = feateng.feateng1(training_data,test_data,add_sessions,rm_classes,merge_classes,debug)

    #print some information about the data
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

    #split training data into train and holdout sets (last 10% of data)
    X,Y = split_train_test(df_train,labels)

    #If requested, rescale predictors between zero and one.
    X = rescale_predictors(X)

    #format data for fitting
    #some counts
    n_classes = np.shape(np.unique(Y['train']))[0]
    n_feats = np.shape(X['train'])[1]
    input_dim = n_feats
    #format the data for fitting
    x_train = np.array(X['train'])
    y_train = keras.utils.to_categorical(Y['train'], num_classes=12)
    
    #set up randomized search parameter distributions
    import scipy.stats as st
    params = {
        "batch_size": eval(batch_size),
        "epochs": eval(epochs),
        "learn_rate": eval(learn_rate),
        "momentum": eval(momentum),
        "init_mode": eval(init_mode),
        "activation": eval(activation),
        "dropout_rate": eval(dropout_rate),
        "neurons": eval(neurons),
        "layers": eval(layers),
    }
    # Function to create model for KerasClassifier
    
    def create_model(input_dim=input_dim, n_classes=n_classes, learn_rate=0.01, momentum=0,
                     init_mode='uniform',activation='relu', dropout_rate=0.0, neurons=1,
                     layers=1.0):
        #set up the network architecture
        #   Dense(n) is a fully-connected layer with n hidden units.
        #   Note that the number of units in the first layer must be at
        #   least as large as the number of features.
        #   In the first layer, you must specify the expected input data shape:
        #   here, n_feats-dimensional vectors.
        model = Sequential()
        model.add(Dense(neurons, activation=activation, input_dim=input_dim, kernel_initializer=init_mode))
        model.add(Dropout(dropout_rate))
        l = 0
        while l<layers:
            model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode))
            model.add(Dropout(dropout_rate))
            l = l+1
        model.add(Dense(n_classes, activation='softmax', kernel_initializer=init_mode))
        #compile the model
        sgd = SGD(lr=learn_rate, decay=1e-6, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model
    # fix the random seed for reproducibility, but note that
    # in practice we likely want to run the model with multiple initial states.
    np.random.seed(eval(seed))
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=1)
    #execute randomized search
    tic = time.time()
    from sklearn.model_selection import RandomizedSearchCV
    gs = RandomizedSearchCV(model, params, n_jobs=1, n_iter=eval(n_iter), verbose=2, cv=cv)
    gs_result = gs.fit(x_train, y_train)
    toc = time.time()-tic
    # summarize results
    print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
    means = gs_result.cv_results_['mean_test_score']
    stds = gs_result.cv_results_['std_test_score']
    params = gs_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Total run time %4.4f seconds") % (toc)
    #write results to txt file
    print "\tWriting to %s..." % (output_filename)
    f = open(output_filename,'a')
    for mean, stdev, param in zip(means, stds, params):
        f.write("%f,%f,%f,%f,%r\n" % (mean, stdev, eval(seed), toc, param))
    f.close()
    print "\tDONE."


if __name__ == '__main__':

    #set local_model to True for debugging on local machine
    local_mode = False
    if local_mode:
        print "Running in local mode."
        data_directory = "../data/"
        output_filename = "fitKeras_output.txt"
        seed = "0"
        n_iter = "1"
        cv = "5"
        batch_size = "[10,20]"
        epochs = "[5,10]"
        learn_rate = "[0.1,0.2]"
        momentum = "[0,0.01]"
        init_mode = "['uniform','lecun_uniform']"
        activation = "['relu','sigmoid']"
        dropout_rate = "[0.2]"
        neurons = "st.randint(n_feats, 200)"
        layers = "[1.0]"
    #if running on cluster, then read environmental variables here
    if not local_mode:
        data_directory = 'data/'
        output_filename = os.environ['OUTPUT_FILENAME']
        seed = os.environ['SEED']
        n_iter = os.environ['N_ITER']
        cv = os.environ['CV']
        batch_size = os.environ['BATCH_SIZE']
        epochs = os.environ['EPOCHS']
        learn_rate = os.environ['LEARN_RATE']
        momentum = os.environ['MOMENTUM']
        init_mode = os.environ['INIT_MODE']
        activation = os.environ['ACTIVATION']
        dropout_rate = os.environ['DROPOUT_RATE']
        neurons = os.environ['NEURONS']
        layers = os.environ['LAYERS']

    #print the inputs
    print "Input variables:"
    print "\t data_directory = %s" % (data_directory)
    print "\t output_filename = %s" % (output_filename)
    print "\t seed = %s" % (seed)
    print "\t n_iter = %s" % (n_iter)
    print "\t cv = %s" % (cv)
    print "\t batch_size = %s" % (batch_size)
    print "\t epochs = %s" % (epochs)
    print "\t learn_rate = %s" % (learn_rate)
    print "\t momentum = %s" % (momentum)
    print "\t init_mode = %s" % (init_mode)
    print "\t activation = %s" % (activation)
    print "\t dropout_rate = %s" % (dropout_rate)
    print "\t neurons = %s" % (neurons)
    print "\t layers = %s" % (layers)

    #execute main function
    main(data_directory, output_filename, seed, n_iter, cv, batch_size, epochs, learn_rate, momentum, init_mode, activation, neurons, layers)