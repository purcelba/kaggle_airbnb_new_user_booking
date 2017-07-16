import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from airbnb import feateng
from sklearn.preprocessing import LabelEncoder, label_binarize, MinMaxScaler
import h5py
"""
Fit a multilayer perceptron model implemented in Keras with a fixed set of hyperparameters
to the airbnb data.

If running on HPC computing cluster, must import the following modules:
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

def main(data_directory,add_sessions,output_file_name,rescale_predictors,seed,batch_size,epochs,learn_rate,momentum,init_mode,
               activation,dropout_rate,neurons,layers):
    """
    Fit multilayer perceptron implemented in Keras to the kaggle-airbnb dataset with a 
    specified set of hyperparameters.  The fitted model is saved in h5 format to be loaded
    later.
    
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
        - seed, int, fix the PRNG seed
      Hyperparameters:
        - batch_size, int, int, mini-batch size for stochastic gradient descent
        - epochs, int, number of training epochs (i.e., how many times will full data set be used for training)
        - learn_rate, float, learning rate for stochastic gradient descent
        - momentum, float, momentum for stochastic gradient descent (fraction of the  update vector of the past time step to be added to the current vector).
        - init_mode, str, weight initialization (e.g., 'random_uniform' for random).  See https://keras.io/initializers/ for other options. 
        - activation, str, activation function for input and hidden layers (e.g., 'relu' for rectified linear unit).  See https://keras.io/activations/ for other options.
        - neurons, int, number of neurons in the input and hidden layers 
        - dropout_rate, float,  list of float, fraction of input units randomly set to 0 at each update during training time, which helps prevent overfitting.
        - layers, list of int, number of hidden layers.
    
    Returns:
      No variables returned.  Model results are saved as h5 file for reloading.

    """
    # Load the data and generate features
    debug = False
    merge_classes = []
    rm_classes = []
    training_data = data_directory + 'train_users_2.csv'
    test_data = data_directory + 'test_users.csv'
    df_train, df_test, labels = feateng.feateng1(training_data, test_data, add_sessions, rm_classes, merge_classes, rescale_predictors, debug)

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

    # format data for fitting
    # some counts
    n_classes = np.shape(np.unique(Y['train']))[0]
    n_feats = np.shape(X['train'])[1]
    input_dim = n_feats
    # format the data for fitting
    x_train = np.array(X['train'])
    y_train = keras.utils.to_categorical(Y['train'], num_classes=12)

    #set up the model
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_dim=input_dim, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    l = 0
    while l < layers:
        model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode))
        model.add(Dropout(dropout_rate))
        l = l + 1
    model.add(Dense(n_classes, activation='softmax', kernel_initializer=init_mode))
    # compile the model
    sgd = SGD(lr=learn_rate, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    #fit the model
    np.random.seed(seed)
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size)

    #save the model
    model.save(output_file_name)

if __name__ == '__main__':
    #choose params
    add_sessions = 'bin'
    data_directory = 'data/'
    output_file_name = 'kerasModel6.h5'
    rescale_predictors = True
    seed = 88
    batch_size = 100
    epochs = 500
    learn_rate = 0.001
    momentum = 0.5
    init_mode = 'lecun_uniform'
    activation = 'relu'
    dropout_rate = 0.1
    neurons = 118
    layers = 1

    main(data_directory, add_sessions, output_file_name, seed, batch_size, epochs, learn_rate, momentum, init_mode,
         activation, dropout_rate, neurons, layers)