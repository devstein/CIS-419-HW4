'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn import preprocessing

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=2.0, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs

        self.thetas = {}
        self.nodes = {}
        self.deltas = {}
        self.gradients = {}

        np.random.seed(42)

        self.lamda = 0.001


    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))


    def forwardPropagation(self, X):
        n, d = X.shape

        self.nodes[0] = np.c_[np.ones(n), X]

        for i in range(self.L - 1):
            tn = self.sigmoid(self.nodes[i].dot(self.thetas[i+1].T))
            self.nodes[i+1] = np.c_[np.zeros(n), tn]

        self.nodes[self.L - 1] = self.nodes[self.L - 1][:, 1:]

    def backPropagation(self, y):
        self.deltas[self.L - 1] = self.nodes[self.L - 1] - y

        for i in reversed(range(self.L - 1)):
            node = self.nodes[i]
            theta = self.thetas[i+1]
            delta = self.deltas[i+1]

            n, d = theta.shape

            self.deltas[i] = np.multiply(np.dot(delta, theta), np.multiply(node, (1 - node)))[:, 1:]

            gradient = np.dot(self.deltas[i+1].T, self.nodes[i]) / len(y)
            d = np.append(np.zeros((n, 1)), theta[:, 1:], axis = 1)
            self.gradients[i+1] = gradient + d * self.lamda
            self.thetas[i+1] = theta - self.learningRate * self.gradients[i+1]


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n, d = X.shape

        unique_y = np.unique(y)

        b_y = preprocessing.label_binarize(y, unique_y)

        # self.layers_info = np.array([d, self.layers, unique_y.size])
        # self.L = self.layers_info.size
        self.L = 3

        self.thetas[1] = np.random.uniform(-self.epsilon, self.epsilon,
                        (np.squeeze(self.layers), d + 1))
        self.thetas[2] = np.random.uniform(-self.epsilon, self.epsilon,
                        (unique_y.size, np.squeeze(self.layers) + 1))

        # Gradient Descent
        for _ in range(self.numEpochs):
            self.forwardPropagation(X)
            self.backPropagation(b_y)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        self.forwardPropagation(X)
        pred = np.argmax(self.nodes[self.L - 1], axis = 1)
        return pred


    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
