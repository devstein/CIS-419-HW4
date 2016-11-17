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
        self.errors = {}
        self.lambdaVal = 0.001
        np.random.seed(42)


    def sigmoid(self, Z):
        return 1.0/ (1.0 + np.exp(-Z))

    def forwardPropagate(self, X, thetas):
        #neuron acitvation
        #neuron transfer -> sigmoid function
        #forward propogate 
        n,d = X.shape

        X = np.c_[np.ones(n), X]

        self.inputs[0] = X

        for i in range(self.numLayers - 1):
            z = self.inputs[i].dot(self.thetas[i+1].T)
            sig = self.sigmoid(z)
            self.inputs[i+1] = np.c_[np.zeros(n), sig]

        self.inputs[self.numLayers - 1] = self.inputs[self.numLayers - 1][:, 1:]



    def calculateGradient(self, X, y):

        gradients = {}

        for i in reversed(range(self.numLayers - 1)):
            n, d = self.thetas[i+1].shape
            l, w = X.shape

            gradients[i+1] = np.dot(self.errors[i+1].T, self.inputs[i])
            regularized = np.concatenate((np.zeros([n, 1]), self.thetas[i+1][:,1:]), axis = 1) * self.lambdaVal
            gradients[i+1] = (gradients[i+1]/ l + regularized) 
            self.thetas[i+1] = self.thetas[i+1] - self.learningRate * gradients[i+1]


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

        for epoch in range(self.numEpochs):
            #forward propogate
            self.forwardPropagate(X, self.thetas)
            #do back propogation 
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
        