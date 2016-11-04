import tensorflow as tf
import numpy as np
import operator

class neuralnetwork(object):
    def __init__(self, numfeatures, num_nodes=4,learning_rate=0.1,keep_prob=1):

        # create session
        self.keep_prop = keep_prob
        self.sess = tf.Session()

        # create placeholders for inputs
        self.x = self.placeholder([None, numfeatures],tf.float32,"state")
        self.y = self.placeholder([None, 1], tf.float32, "state")

        # weight and bias variables for neural network
        self.w1 = self.weight_variable([numfeatures, num_nodes],'w1')
        self.b1 = self.bias_variable([num_nodes],'b1')
        self.w2 = self.weight_variable([num_nodes, num_nodes],'w2')
        self.b2 = self.bias_variable([1],'b2')
        self.w3 = self.weight_variable([num_nodes, 1],'w3')
        self.b3 = self.bias_variable([1],'b3')

        # create model
        y = self.forwardpropogate(self.x)

         # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, self.y))
        self.update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # initialize all variables
        self.sess.run([tf.initialize_all_variables()])


    def forwardpropogate(self,x):
        layer1 = tf.nn.relu(tf.matmul(x, self.w1)) + self.b1;
        layer1drop = tf.nn.dropout(layer1,keep_prob=self.keep_prop)

        layer2 = tf.nn.relu(tf.matmul(layer1drop, self.w2)) + self.b2;
        layer2drop = tf.nn.dropout(layer2, keep_prob=self.keep_prop)

        y = tf.nn.relu(tf.matmul(layer2drop, self.w3)) + self.b3;
        return y


    def train(self,x,y):
        self.sess.run([self.update], feed_dict={
            self.x: np.array(x),
            self.y: np.array(y)
        })

    def predict(self,testx):
        return self.forwardpropogate(testx)


    def placeholder(self,shape,type,name):
        return tf.placeholder(dtype=type, shape=shape, name=name)

    def weight_variable(self,shape,name):
        return tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self,shape,name):
        return tf.get_variable(name,shape=shape)

if __name__ == "__main__":
    pass