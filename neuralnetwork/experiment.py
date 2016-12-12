import numpy as np
from neuralnetwork import NeuralNetwork
from perceptron import Perceptron
from random import shuffle
import scipy.io as sio
import os


if __name__ == "__main__":

    # load path
    data_path = '/Users/rafihaque/machinelearningproject/neuralnetwork/'
    data = sio.loadmat(data_path+'transformed_data_2.mat')
    data_struct = data['newdata']
    x = data_struct['features'][0][0]
    y = data_struct['survival'][0][0]
    save = 'results.mat'

    # paramters
    num_nodes = 5
    learn_rate = 0.1

    cv = 0.8
    num_folds  = 5
    num_obs = x.shape[0]
    num_feats  = x.shape[1]
    num_tr_obs = int(round(cv * num_obs))
    num_te_obs = num_obs-num_tr_obs-1

    # outputs
    yp_train = np.zeros((num_folds,num_tr_obs))
    yp_test  = np.zeros((num_folds,num_te_obs))
    yn_train = np.zeros((num_folds, num_nodes, num_tr_obs))
    yn_test  = np.zeros((num_folds, num_nodes, num_te_obs))
    y_train  = np.zeros((num_folds,num_tr_obs))
    y_test   = np.zeros((num_folds,num_te_obs))

    for fold in range(num_folds):
        print "FOLD:", fold
        # randomize observation indices
        all_obs = range(num_obs)
        shuffle(all_obs)
        train_obs = all_obs[:num_tr_obs]
        test_obs = all_obs[num_tr_obs + 1:]

        # store ytrain and ytest
        labels = np.argmax(y, axis=1)
        y_train[fold] = labels[train_obs]
        y_test[fold] = labels[test_obs]

        # train and test perceptron
        perc = Perceptron(num_feats=num_feats,learn_rate=learn_rate)
        perc.train(x[train_obs],y[train_obs])
        yp_train[fold] = np.argmax(perc.predict(x[train_obs]), axis=1)
        yp_test[fold] = np.argmax(perc.predict(x[test_obs]), axis=1)
        perc.close()

        for node in range(num_nodes):
            print "NODE:",3*node + 1

            # train and test neural network
            nn = NeuralNetwork(num_feats=num_feats, num_nodes=3*node + 1, learn_rate=learn_rate)
            nn.train(x[train_obs],y_train[fold].reshape(num_tr_obs,1))
            yn_train[fold,node] = nn.predict(x[train_obs]).reshape(num_tr_obs, )
            yn_test[fold,node] = nn.predict(x[test_obs]).reshape(num_te_obs,)
            nn.close()



    test = {}
    test['y_train'] = y_train
    test['y_test'] = y_test
    test['yp_train'] = yp_train
    test['yp_test'] = yp_test
    test['yn_train'] = yn_train
    test['yn_test'] = yn_train


    sio.savemat(data_path + save,test)
