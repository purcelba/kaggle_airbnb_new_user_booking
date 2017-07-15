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


def main(data_directory,add_sessions,output_file_name,seed,batch_size,epochs,learn_rate,momentum,init_mode,
               activation,dropout_rate,neurons,layers):
    """
    Load the airbnb data set. Fit it with a keras MLP model with specified parameters. Save the result.

    Params:

    Returns:

    """
    # Load the data and generate features
    debug = False
    merge_classes = []
    rm_classes = []
    training_data = data_directory + 'train_users_2.csv'
    test_data = data_directory + 'test_users.csv'
    df_train, df_test, labels = feateng.feateng1(training_data, test_data, add_sessions, rm_classes, merge_classes, debug)

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

    #kerasModel4
    #seed=95, layers=1, learn_rate=0.0001, dropout_rate=0.1, activation=relu, batch_size=10, epochs=1000, neurons=118, init_mode=lecun_uniform, momentum=0.95 
    
    #kerasModel5
    #same
    
    #kerasModel6
    #seed=88{'layers': 1, 'learn_rate': 0.001, 'dropout_rate': 0.1, 'activation': 'relu', 'batch_size': 100, 'epochs': 500, 'neurons': 118, 'init_mode': 'lecun_uniform', 'momentum': 0.5}


    #choose params
    add_sessions = 'bin'
    data_directory = 'data/'
    output_file_name = 'kerasModel6.h5'
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