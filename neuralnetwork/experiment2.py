import numpy as np
from neuralnetwork import NeuralNetwork
from random import shuffle
import scipy.io as sio
import os
import tensorflow as tf
from sklearn import metrics

if __name__ == "__main__":

    # load data
    dataPath = '/Users/rafihaque/PycharmProjects/machinelearningproject/neuralnetwork/'
    file = 'TransformedModel.mat'
    save = 'Results.mat'
    data = sio.loadmat(dataPath + file)
    data_struct = data['newdata']
    x = data_struct['features'][0][0]
    y = data_struct['survival'][0][0]

    # paramters

    num_nodes = 10
    learn_rate = 0.001
    keep_prob = 1
    cv    = 0.8
    num_folds  = 10
    num_obs    = x.shape[0]
    num_feats  = x.shape[1]
    num_tr_obs = int(round(cv * num_obs))
    num_te_obs = num_obs-num_tr_obs

    # outputs
    y_pred  = np.zeros((num_folds,num_nodes,num_tr_obs))
    y_train = np.zeros((num_folds,num_nodes,num_tr_obs))
    y_test  = np.zeros((num_nodes,num_te_obs))

    for fold in range(num_folds):
        for node in range(num_nodes):

            print "NODE:",node
            print "FOLD:",fold

            # randomize observation indices
            all_obs = range(num_obs)
            shuffle(all_obs)
            train_obs = all_obs[:num_tr_obs]
            test_obs = all_obs[num_tr_obs+1:]

            # initialize the neural network
            nn = NeuralNetwork(num_feats=num_feats, num_nodes=node, learn_rate=learn_rate, keep_prob=keep_prob)

            # store ytrain and ytest
            print y[train_obs][0]
            print ytrain[fold,node,0].shape[1]
            y_train[fold,node,0] = y[train_obs]
            y_test[fold,node,:] =  y[test_obs]

            # train the neural network
            print x[train_obs]
            nn.train(x[train_obs],y[train_obs])

            # test the neural network
            ypred[node,fold] = nn.predict(x[test_obs])


            # fpr, tpr = metrics.roc_curve(y[allind[ind+1:]],yhat,pos_label=1)
            # print metrics.auc(fpr,tpr)


    test = {}
    test['ytrain'] = ytrain
    test['ytest'] = ytest
    test['ypred'] = ypred
    sio.savemat(dataPath + save,test)