#test file

import numpy as np
from nn import NeuralNet

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


# load the data
filePathX = "data/digitsX.dat"
file = open(filePathX,'r')
allDataX = np.loadtxt(file, delimiter=',')

X = allDataX

filePathY = "data/digitsY.dat"
file = open(filePathY,'r')
allDataY = np.loadtxt(file)
y = allDataY

# print "X Data Shape:",X.shape
# print "Y Data shape:",y.shape

# X = np.array([[1,2,3,4,5,6,7], [4,4,4,5,5,5,6], [9,8,7,6,5,4,3], [4,1,9,5,8,3,6]])
# X = np.array([[1,2,3,4,5], [4,4,4,5,5], [9,8,7,6,5], [4,8,9,2,3], [7,2,6,3,8]])
# y = np.array([2,5,8,2,7])
layers = np.array([25])

modelNets = NeuralNet(layers, learningRate = 3, numEpochs = 1000, epsilon = 0.5)
modelNets.fit(X, y)

# output predictions on the remaining data
ypred_Nets = modelNets.predict(X)

# compute the training accuracy of the model
accuracyNets = accuracy_score(y, ypred_Nets)

print "Neural Nets Accuracy = "+str(accuracyNets)

filen = "Hidden_Layers.bmp"
modelNets.visualizeHiddenNodes(filen)