import numpy as np
from neuralnetwork import NeuralNetwork
from random import shuffle
import scipy as sio
import os

if __name__ == "__main__":


    # stores all mat files in the datapath in a list
    dataPath = ""
    savePath = ""
    mats = []
    for file in os.listdir(dataPath):
        mats.append(sio.loadmat(dataPath + file))

    # assign data using key
    x = np.array([mats[0]["x"]])
    y = np.array([mats[0]["y"]])

    # paramters
    folds = 5
    nodes = 4
    learning_rate = 0.1
    keep_prop = 1
    cvperc = 4/5


    for fold in folds:
        # get shuffled indices
        range = shuffle((0, x.shape[0]))

        # initialize the neural network
        nn = NeuralNetwork(numfeatures=x.shape[1],
                           num_nodes=nodes,
                           learning_rate=learning_rate,
                            keep_prob=keep_prop)


        # train the neural network
        ind =  cvperc*len(range)
        nn.train(x[range[:ind]],y[range[:ind]])

        # test the neural network
        nn.predict(x[range[ind+1:]],y[range[ind+1:]])

        # save out model
        nn.savemodel(savePath)