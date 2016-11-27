import tensorflow as tf
import numpy as np
import operator
import scipy.io as sio

class NeuralNetwork(object):
    def __init__(self, num_feats, num_nodes=4,learn_rate=0.001,keep_prob=1):

        # create session
        self.keep_prop = keep_prob
        self.sess = tf.Session()

        # create placeholders for inputs
        self.x = self.placeholder('x',tf.float32,[None, num_feats])
        self.y = self.placeholder('y',tf.float32,[None, 1])

        # weight and bias variables for neural network
        self.w1 = self.weight_variable('w1',tf.float32,[num_feats, num_nodes])
        self.w2 = self.weight_variable('w2',tf.float32,[num_nodes, num_nodes])
        self.w3 = self.weight_variable('w3',tf.float32,[num_nodes, 1])

        # create model
        self.yhat = self.model(self.x)

         # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.yhat, self.y))
        self.update = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.loss)

        # initialize all variables
        self.sess.run([tf.initialize_all_variables()])
        self.saver = tf.train.Saver()

    def model(self,x):
        layer1 = tf.nn.sigmoid(tf.matmul(self.x, self.w1))
        # layer1drop = tf.nn.dropout(layer1,self.keep_prop)
        layer2 = tf.nn.sigmoid(tf.matmul(layer1, self.w2))
        # layer2drop = tf.nn.dropout(layer2, self.keep_prop)
        layer2drop = tf.nn.dropout(layer2, self.keep_prop)
        return tf.nn.sigmoid(tf.matmul(layer2,self.w3))

    def train(self,x,y):
        self.sess.run([self.update], feed_dict={
            self.x: x,
            self.y: y
        })

    def predict(self,testx):
        return self.sess.run(self.yhat, feed_dict={
            self.x: testx
        })

    def savemodel(self,path):
        self.saver.save(self.sess,path)

    def close(self):
        self.sess.close()

    def placeholder(self,name,type,shape):
        return tf.placeholder(name=name,dtype=type,shape=shape)

    def weight_variable(self,name,type,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name='w1')




if __name__ == "__main__":
    pass