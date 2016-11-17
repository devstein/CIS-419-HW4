'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn import preprocessing

# import sys
# sys.path.append("/usr/local/lib/python2.7/site-packages")

class NeuralNet:

    def __init__(self, layers, learningRate = 2.0, epsilon = 0.12, numEpochs = 100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers                    # Number of nodes in each Hidden Layer.
        self.epsilon = epsilon                  # One half the interval around zero for the initial weights
        self.learningRate = learningRate        # Learning rate for Back-Propagation
        self.numEpochs = numEpochs              # Number of Epochs for Back-Propagation
        self.a_vals = None                      # Stores all the a vlaues for each layer
        self.lambda_ = 0.001                    # Lambda for regularization
        self.thetas = None                      # Keep my thetas for predictions
        self.classes_ = None                    # Keep all the unique classes


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        verbose = 0
        do_backprop = 1
        if verbose:
            print "============================================="
            print "=========== Neural Nets Testing ============="
            print "============================================="
            print ""
        (n, d) = X.shape

        # Number of Classes. Unique values in y vector. Output layer size.
        clf = preprocessing.LabelBinarizer()
        clf.fit(y)
        self.classes = clf.classes_
        yd = clf.classes_.size

        # Binarize y
        y_binar = clf.transform(y)
        # print "Y Binarized shape:", y_binar.shape
        # print "First instance of Y Binarized:", y_binar[0,:]
        # print "Last Instance of Y Binarized:", y_binar[-1,:]

        # Create a dictionary of self.thetas for the layers
        self.thetas = dict()

        # Initialize all self.thetas randonmly
        # If we have at least one hidden layer
        if(1): #self.layers.size > 0):
          # Add the input and output layer to the lays vector
          lays    = np.concatenate((self.layers, [yd]), axis=1)         # Adds output layer size to the lays vector
          lays    = np.concatenate(([d], lays), axis = 1)
          
          
          # Does the same for all the other layers. From 1st hidden to output
          for i in range(lays.size - 1):
            # Thetas as numpy arrays stored in a dictionary
            self.thetas[i+1] = np.random.random_sample([lays[i+1], lays[i]+1]) * 2.0 * self.epsilon - self.epsilon
            # theta = np.concatenate((theta, theta_i.ravel()), axis=1)      # This was for the 1 by n long vector we never used
         
        else:
           self.thetas[1] =  np.random.random_sample([d+1, yd]) * 2 * self.epsilon - self.epsilon
        
        last_layer = len(self.thetas)    # Index in my last layer

        # Display some information
        if verbose:
            print "Number of Instances:", X.shape[0]
            print "Number of Classes:", yd
            print "Number of Features", X.shape[1]
            print ""
            print "Layers Including input and output ones:", lays
            print "Last Layer Index:",last_layer
            for i in range(len(self.thetas)):
                print "Theta",i+1, "size:", self.thetas[i+1].shape
                # print self.thetas[i+1][0:4,:]
            print ""
            print "Starting Back-Propagation"
                


#=======================================================================================================#
        # Backpropagation
#=======================================================================================================#
        
        if do_backprop:
            deltas = dir()          # Create deltas dictionaty to store them
            Gradient = dir()

            for e in range(self.numEpochs):
                # print "======= NEW EPOCH ======="
                # Compute Forward Propagation
                self.fow_prop(X, self.thetas)

                # Compute Deltas
                deltas[last_layer] = self.a_vals[last_layer][:,1:] - y_binar
                # print "Shape of Deltas in last layer (",last_layer,"), (classes layer):", deltas[last_layer].shape
            
                # Implementation of Slide 39 in Neural Nets
                for i in range(len(self.thetas)-1,-1,-1):
                    # print "Index for delta calculation:",i
                    j = i+1
                    if(i<>0):
                        # In a couple of places we have to take off first row with bias terms
                        sigmoid_gradient = np.multiply(self.a_vals[i][:,1:], (1.0 - self.a_vals[i][:,1:]))
                        deltas[i] =  np.multiply(np.dot(deltas[j],self.thetas[j][:,1:]), sigmoid_gradient)
                        # print "Shape of Delta ",i,":",deltas[i].shape

            

                for i in range(len(self.thetas)):
                    # Gradient[j] = 0
                    j = i+1     # Because this starts in 0
                    # print j
                    # Calculate the Gradient
                    Gradient[j] = np.dot(deltas[j].T, self.a_vals[i])
                    # Add regularization to all minus the first column
                    regu = np.concatenate((np.zeros([self.thetas[j].shape[0], 1]), self.thetas[j][:,1:]), axis = 1) * self.lambda_
                    Gradient[j] = (Gradient[j]/ X.shape[0] + regu) 
                    self.thetas[j] = self.thetas[j] - self.learningRate * Gradient[j]

        

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        self.fow_prop(X, self.thetas)
        last = len(self.thetas)
        # It seems that the features of the last layer shouldn't add up to 1 in the end
        # print "Check that Probabilities by instances add up to 1"
        # print self.a_vals[last][0:3,1:]
        y_pred = np.argmax(self.a_vals[last][:,1:], axis = 1)
        # print "Prediction Shape:", y_pred.shape
        return y_pred
    
    
    def visualizeHiddenNodes(self, filen):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''


    def fow_prop(self, X, thetas):
        '''
        Created by Francisco de Villalobos
        Arguments:
            X is a n-by-d numpy array
            thetas is a dictionary of theta matrices
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        # Add column of 1's to X so that we have all the bias terms
        X = np.c_[np.ones(X.shape[0]), X]
        n,d = X.shape
        
        # Create dictionary of a's
        self.a_vals = dict()

        # Rows will be the instances
        # Columns will be the a values for that hidden layer
        self.a_vals[0] =  X
        # print self.a_vals[0].shape
        self.a_vals[1] = self.sigmoid(np.dot(X, thetas[1].T))                           # g_z
        self.a_vals[1] = np.c_[np.ones(self.a_vals[1].shape[0]), self.a_vals[1]]        # Add bias to hidden layer
        # print "Theta 1 shape:", thetas[1].shape, " Input Layer Shape:   ",X.shape," A 1 shape:", self.a_vals[1].shape
        for i in range(len(thetas)-1):
            self.a_vals[i+2] = self.sigmoid(np.dot(self.a_vals[i+1], thetas[i+2].T))
            # if (i < len(thetas) - 2):       # Do not add bias to the last layer. Only classes here.
            self.a_vals[i+2] = np.c_[np.ones(self.a_vals[i+2].shape[0]), self.a_vals[i+2]]
            # print "Theta",i+2,"shape:", thetas[i+2].shape, " Hidden Layer",i+1,"shape:", self.a_vals[i+1].shape," A",i+2,"shape:", self.a_vals[i+2].shape

        # Checked that the a values in the last layer are all between 0 and 1.
        # print self.a_vals[len(thetas)][0,:]
    


    def sigmoid(self, z):
		    '''
		    Used to calculate the sigmoid function
		    of either numbers or matrices
		    '''
		    return 1.0 / (1.0 + np.exp(-z))

