import numpy as np
from neuralnetwork import NeuralNetwork
from random import shuffle
import scipy.io as sio
import os
import tensorflow as tf

if __name__ == "__main__":


    # load data
    dataPath = '/Users/rafihaque/PycharmProjects/machinelearningproject/'
    file = 'BasicModel.mat'
    data = sio.loadmat(dataPath + file)
    data_struct = data['BasicModel']
    x = np.transpose(data_struct['Features'][0][0])
    s = np.transpose(data_struct['Survival'][0][0])
    c = np.transpose(data_struct['Censored'][0][0])

    # # paramters
    folds = 5
    nodes = 4
    learning_rate = 0.1
    keep_prop = 1
    cvperc = 4/5

    print len(x[1])
    nn = NeuralNetwork(num_features=len(x[0]),
                        num_nodes=nodes,
                        learning_rate=learning_rate,
                        keep_prob=keep_prop)


    nn.train(x,s)
    yhat =  np.array(nn.predict(x))



    # print nn.yhat
    # print nn.w1
    # print nn.x
    #
    #
    # for fold in folds:
    #     # get shuffled indices
    #     range = shuffle((0, x.shape[0]))
    #
    #     # initialize the neural network
    #     nn = NeuralNetwork(numfeatures=x.shape[1],
    #                        num_nodes=nodes,
    #                        learning_rate=learning_rate,
    #                         keep_prob=keep_prop)
    #
    #
    #     # train the neural network
    #     ind =  cvperc*len(range)
    #     nn.train(x[range[:ind]],y[range[:ind]])
    #
    #     # test the neural network
    #     yhat = nn.predict(x[range[ind+1:]],y[range[ind+1:]])
    #
    #     # save out model
    #     nn.savemodel(savePath)