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

    # # paramters
    folds = 5
    nodes = 4
    learning_rate = 0.001
    keep_prop = 1
    cvperc = 0.8



    ytest = []
    ytrain = []
    ypred = []


    for fold in range(folds):
        print "FOLD",fold
        # get shuffled indices

        allind = range(x.shape[0])
        shuffle(allind)

        # initialize the neural network
        nn = NeuralNetwork(num_features=x.shape[1],
                           num_nodes=nodes,
                           learning_rate=learning_rate,
                            keep_prob=keep_prop)

        # store ytrain and ytest
        ind = int(round(cvperc * len(allind)))
        ytrain.append(y[allind[:ind]])
        ytest.append(y[allind[ind+1:]])

        # train the neural network
        nn.train(x[allind[:ind]],y[allind[:ind]])

        # test the neural network
        yhat = nn.predict(x[allind[ind + 1:]])
        ypred.append(yhat)


        # fpr, tpr = metrics.roc_curve(y[allind[ind+1:]],yhat,pos_label=1)
        # print metrics.auc(fpr,tpr)


    test = {}
    test['ytrain'] = ytrain
    test['ytest'] = ytest
    test['ypred'] = ypred
    sio.savemat(dataPath + save,test)