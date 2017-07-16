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
import scipy.stats as st

"""
Randomized search for multilayer perceptron hyperparameters implemented in Keras fitted
to the kaggle-airbnb dataset.  
Inputs are distributions from scipy.stats or sets of MLP hyperparameters from which to samples.
When running as script, setting local_mode to false will read in environmental variables
as input to be used on computing cluster.  
The resulting best fit hyperparameters are output to a text file where they can be loaded 
for further analysis.
"""
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

def main(data_directory, add_sessions, output_filename, rescale_predictors, seed, n_iter, cv, batch_size, epochs, learn_rate, momentum, init_mode, activation, dropout_rate, neurons, layers):
    """
    Randomized search for multilayer perceptron hyperparameters implemented in Keras fitted
    to the kaggle-airbnb dataset.  The resulting best fit hyperparameters are printed to a
    text file where they can be loaded for further analysis.
    
    Params:
      General params:
        - data_diretory, str, path to directory in which data are stored
        - add_sessions, str, can be 'none','bin', 'count', or 'secs'
            - 'none' will not add any sessions features
            - 'bin' will add binary features indicating 1 if a a user took an action was taken and 0 otherwise
            - 'count' will add integer features indicating the number of times a user took an action
            - 'secs' will add integer features indicating the number of seconds users spent on each action
        - output_filename, str, name of txt file in which to write results.
        - rescale_predictors, logical, if True, then will rescale predictors in the feature matrix between 0 and 1.
        - seed, int, set the PRNG seed
        - n_iter, int, number of samples to take from the hyperparameter distributions  
        - cv, int, number of folds for cross validation
      Hyperparameter sets or distributions from which samples will be drawn:
        - batch_size, list of int, mini-batch size for stochastic gradient descent
        - epochs, list of int, number of training epochs (i.e., how many times will full data set be used for training)
        - learn_rate, list of float, learning rate for stochastic gradient descent
        - momentum, list of float, momentum for stochastic gradient descent (fraction of the  update vector of the past time step to be added to the current vector).
        - init_mode, list of str, weight initialization (e.g., 'random_uniform' for random).  See https://keras.io/initializers/ for other options. 
        - activation, list of str, activation function for input and hidden layers (e.g., 'relu' for rectified linear unit).  See https://keras.io/activations/ for other options.
        - neurons, list of int or scipy.stats distribution function, number of neurons in the input and hidden layers 
        - layers, list of int, number of hidden layers.
    
    Returns:
      No variables returned.  Results are printed to txt file defined by output_filename for combining with output of other jobs.

    """
    #Load the data and generate features
    debug = False
    merge_classes = []
    rm_classes = []
    training_data = data_directory + 'train_users_2.csv'
    test_data = data_directory + 'test_users.csv'
    df_train, df_test, labels, id_train, id_test = feateng.feateng1(training_data,test_data,add_sessions,rm_classes,merge_classes,rescale_predictors,debug)

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

    #format data for fitting
    #some counts
    n_classes = np.shape(np.unique(Y['train']))[0]
    n_feats = np.shape(X['train'])[1]
    input_dim = n_feats
    #format the data for fitting
    x_train = np.array(X['train'])
    y_train = keras.utils.to_categorical(Y['train'], num_classes=12)
    
    #set up randomized search parameter distributions
    params = {
        "batch_size": batch_size,
        "epochs": epochs,
        "learn_rate": learn_rate,
        "momentum": momentum,
        "init_mode": init_mode,
        "activation": activation,
        "dropout_rate": dropout_rate,
        "neurons": neurons,
        "layers": layers,
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
    np.random.seed(seed)
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=1)
    #execute randomized search
    tic = time.time()
    from sklearn.model_selection import RandomizedSearchCV
    gs = RandomizedSearchCV(model, params, n_jobs=1, n_iter=n_iter, verbose=2, cv=cv)
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
        f.write("%f,%f,%f,%f,%r\n" % (mean, stdev, seed, toc, param))
    f.close()
    print "\tDONE."


if __name__ == '__main__':

    #set local_model to True for debugging on local machine
    local_mode = True
    if local_mode:
        data_directory = "../data/"
        add_sessions = 'none'
        output_filename = 'fitKeras_output.txt'
        rescale_predictors = True
        seed = 0
        n_iter = 1
        cv = 5
        batch_size = [10,20]
        epochs = [5,10]
        learn_rate = [0.1,0.2]
        momentum = [0,0.01]
        init_mode = ['uniform','lecun_uniform']
        activation = ['relu','sigmoid']
        dropout_rate = [0.2]
        neurons = st.randint(100, 200)
        layers = [1.0]
        print "Running in local mode."
        
        
    #if running on cluster, then read environmental variables here
    if not local_mode:
        #read in environmental variables
        data_directory = 'data/'
        add_sessions = os.environ['ADD_SESSIONS']
        output_filename = os.environ['OUTPUT_FILENAME']
        rescale_predictors = os.environ['RESCALE_PREDICTORS']
        seed = eval(os.environ['SEED'])
        n_iter = eval(os.environ['N_ITER'])
        cv = eval(os.environ['CV'])
        batch_size = eval(os.environ['BATCH_SIZE'])
        epochs = eval(os.environ['EPOCHS'])
        learn_rate = eval(os.environ['LEARN_RATE'])
        momentum = eval(os.environ['MOMENTUM'])
        init_mode = eval(os.environ['INIT_MODE'])
        activation = eval(os.environ['ACTIVATION'])
        dropout_rate = eval(os.environ['DROPOUT_RATE'])
        neurons = eval(os.environ['NEURONS'])
        layers = eval(os.environ['LAYERS'])

    #print the inputs
    print "Input variables:"
    print "\t data_directory = %s" % (data_directory)
    print "\t add_sessions = %s" % (add_sessions)
    print "\t output_filename = %s" % (output_filename)
    print "\t rescale_predictors = %s" % (rescale_predictors)
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
    main(data_directory, add_sessions, output_filename, rescale_predictors, seed, n_iter, cv, batch_size, epochs, learn_rate, momentum, init_mode, activation, dropout_rate, neurons, layers)