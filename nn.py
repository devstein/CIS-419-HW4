'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn import preprocessing

class NeuralNet:


#Don't change the order of the arguments.  
#Just specify a default value for the learningRate for your neural network.
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
        self.inputs = {}
        self.gradients = {}
        self.errors = {}

        np.random.seed(42)


    def sigmoid(self, Z):
        return 1.0/ (1.0 + np.exp(-Z))

    def forwardPropagate(self, X, thetas):
        #neuron acitvation
        #neuron transfer -> sigmoid function
        #forward propogate 
        n,d = X.shape

        self.inputs[0] = np.c_[np.ones(n), X]

        numLayers = len(thetas)

        for i in range(self.numLayers - 1):
            z = self.sigmoid(self.inputs[i].dot(self.thetas[i+1].T))
            self.inputs[i+1] = np.c_[np.zeros(n), z]

        self.inputs[self.numLayers - 1] = self.inputs[self.numLayers - 1][:, 1:]


        # #set X as first input array
        # self.inputs[0] = X
        # #set next input as sigmoid on the activated neurons 
        # self.inputs[1] = self.sigmoid(np.dot(X, thetas[1].T))
        # #add bias to layer
        # self.inputs[1] = np.c_[np.ones(self.inputs[1].shape[0]), self.inputs[1]]

        # print self.inputs[1][:,16:].shape

        # for i in range(2, numLayers - 1):
        #     self.inputs[i] = self.sigmoid(np.dot(self.inputs[i-1], thetas[i].T))
        #     self.inputs[i] = np.c_[np.ones(self.inputs[i].shape[0]), self.inputs[i]]

        # for i in range(numLayers-1):
        #     self.inputs[i+2] = self.sigmoid(np.dot(self.inputs[i+1], thetas[i+2].T))
        #     # if (i < len(thetas) - 2):       # Do not add bias to the last layer. Only classes here.
        #     self.inputs[i+2] = np.c_[np.ones(self.inputs[i+2].shape[0]), self.inputs[i+2]]

    
    def calculateGradient(self, X, y):

        for i in reversed(range(self.numLayers - 1)):
            n, d = self.thetas[i+1].shape

            gradient = np.dot(self.errors[i+1].T, self.inputs[i]) / len(y)
            z = np.append(np.zeros((n, 1)), self.thetas[i+1][:, 1:], axis = 1)
            self.gradients[i+1] = gradient + z * 0.001
            self.thetas[i+1] = self.thetas[i+1] - self.learningRate * self.gradients[i+1]


    def calculateErrors(self, X, y):
        self.errors[self.numLayers - 1] = self.inputs[self.numLayers - 1] - y
        #loop backwards
        for i in reversed(range(self.numLayers - 1)):
            z1 = np.dot(self.errors[i+1], self.thetas[i+1])
            z2 = np.multiply(self.inputs[i], (1 - self.inputs[i]))
            self.errors[i] = np.multiply(z1, z2)[:, 1:]

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape

        self.classes = np.unique(y)
        numClasses = len(self.classes)

        binarizer = preprocessing.LabelBinarizer()
        binarizer = binarizer.fit(y)
        y_binarized = binarizer.transform(y)

        setLayers = np.concatenate((self.layers, [numClasses])) 
        setLayers = np.concatenate(([d], setLayers))

        self.numLayers = setLayers.size

        for i in range(1, self.numLayers):
            self.thetas[i] = np.random.random_sample([setLayers[i], setLayers[i-1]+1]) * 2.0 * self.epsilon - self.epsilon

        for e in range(self.numEpochs):
            #forward propogate
            self.forwardPropagate(X, self.thetas)

            #compute errors
            self.calculateErrors(X, y_binarized)
            #udapte theta with gradient
            self.calculateGradient(X, y_binarized)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        self.forwardPropagate(X, self.thetas)
        return np.argmax(self.inputs[self.numLayers - 1], axis = 1)

    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        