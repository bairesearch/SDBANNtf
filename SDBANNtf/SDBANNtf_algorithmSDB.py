"""SDBANNtf_algorithmSDB.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SDBANNtf_main.py

# Usage:
see SDBANNtf_main.py

# Description:
SDBANNtf algorithm SDB - define simulated dendritic branch artificial neural network

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs

useDependentSubbranches = False
if(useDependentSubbranches):
	numberOfDependentSubbranches = 3
	numberOfIndependentDendriticBranches = 10
	numberOfIndependentDendriticSubbranchesSplit = [numberOfIndependentDendriticBranches, 2, 1]
	numberOfIndependentDendriticSubbranches = [numberOfIndependentDendriticBranches//numberOfIndependentDendriticSubbranchesSplit[0], numberOfIndependentDendriticBranches//numberOfIndependentDendriticSubbranchesSplit[1], numberOfIndependentDendriticBranches//numberOfIndependentDendriticSubbranchesSplit[2]]	#all values must be factors of numberOfIndependentDendriticBranches, eg 1, 5, 10
else:
	numberOfIndependentDendriticBranches = 10

normaliseActivationSparsity = False
if(normaliseActivationSparsity):
	weightStddev = 0.05	#from https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomNormal

debugNormaliseActivationSparsity = False
debugOnlyTrainFinalLayer = False
debugSingleLayerNetwork = False
debugFastTrain = False
debugSingleLayerNetwork = False

supportMultipleNetworks = True

supportSkipLayers = False

normaliseFirstLayer = False	#consider for normaliseActivationSparsity
equaliseNumberExamplesPerClass = False


W = {}
B = {}
if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
if(supportSkipLayers):
	Ztrace = {}
	Atrace = {}

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0

def defineTrainingParameters(dataset):
	global batchSize
	
	learningRate = 0.001
	if(debugNormaliseActivationSparsity):
		batchSize = 10
	else:
		batchSize = 100
	numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000

	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	if(debugSingleLayerNetwork):
		n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParametersANNsingleLayer(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet)
	else:
		n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=False)
	
	return numberOfLayers
	
	
def defineNetworkParametersANNlegacy(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	datasetNumClasses = num_output_neurons
	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	n_h_0 = n_x
	if(dataset == "POStagSequence"):
		n_h_1 = int(datasetNumFeatures*3) # 1st layer number of neurons.
		n_h_2 = int(datasetNumFeatures/2) # 2nd layer number of neurons.
	elif(dataset == "SmallDataset"):
		n_h_1 = 4
		n_h_2 = 4
	else:
		print("dataset unsupported")
		exit()
	n_h_3 = n_y
	if(debugSingleLayerNetwork):
		n_h = [n_h_0, n_h_3]	
	else:
		n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
	numberOfLayers = len(n_h)-1
	
	print("defineNetworkParametersANNlegacy, n_h = ", n_h)
	
	return n_h, numberOfLayers, numberOfNetworks, datasetNumClasses
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l1 in range(1, numberOfLayers+1):
			if(supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						if(useDependentSubbranches):
							Wlayer = randomNormal([numberOfDependentSubbranches, numberOfIndependentDendriticBranches, n_h[l2], n_h[l1]])
							Wlayer = tf.Variable(initialiseWlayerSubbranchValues(Wlayer))
						else:
							Wlayer = randomNormal([numberOfIndependentDendriticBranches, n_h[l2], n_h[l1]]) 
						W[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "W")] = tf.Variable(Wlayer)
			else:	
				if(useDependentSubbranches):
					Wlayer = tf.Variable(randomNormal([numberOfDependentSubbranches, numberOfIndependentDendriticBranches, n_h[l1-1], n_h[l1]]))
					Wlayer = tf.Variable(initialiseWlayerSubbranchValues(Wlayer))
				else:
					Wlayer = tf.Variable(randomNormal([numberOfIndependentDendriticBranches, n_h[l1-1], n_h[l1]]))
				W[generateParameterNameNetwork(networkIndex, l1, "W")] = Wlayer
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(tf.zeros(n_h[l1]))

			if(supportSkipLayers):
				Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))

			#print("Wlayer = ", W[generateParameterNameNetwork(networkIndex, l1, "W")])

	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer = tf.Variable(WlayerF)
			BlayerF = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer= tf.Variable(BlayerF)	#not currently used
					
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationANN(x, networkIndex)

def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
	return neuralNetworkPropagationANN(x, networkIndex, l)

#if(supportMultipleNetworks):
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred

def initialiseWlayerSubbranchValues(Wlayer):
	WlayerSubbranchAveragedList = []
	for subbranchIndex in range(numberOfDependentSubbranches):	
		WlayerSubbranchAveraged = equaliseWlayerSubbranchValues(subbranchIndex, Wlayer, False)
		WlayerSubbranchAveragedList.append(WlayerSubbranchAveraged)
	WlayerSubbranchAveragedStacked = tf.stack(WlayerSubbranchAveragedList, axis=0)
	return WlayerSubbranchAveragedStacked

def equaliseWlayerSubbranchValues(subbranchIndex, Wlayer, takeAverageOrFirst):
	#equalise WlayerSubbranch values (take average of values after last backprop update)
	WlayerSubbranch = Wlayer[subbranchIndex, :, :, :]
	WlayerSubbranchList = tf.split(WlayerSubbranch, num_or_size_splits=numberOfIndependentDendriticSubbranchesSplit[subbranchIndex], axis=0)
	if(takeAverageOrFirst):
		WlayerSubbranchAveraged = tf.stack(WlayerSubbranchList, axis=0)
		WlayerSubbranchAveraged = tf.reduce_mean(WlayerSubbranchAveraged, axis=0)
	else:
		WlayerSubbranchAveraged = WlayerSubbranchList[0]
	tile = [numberOfIndependentDendriticSubbranchesSplit[subbranchIndex], 1, 1]
	WlayerSubbranchAveraged = tf.tile(WlayerSubbranchAveraged, tile)
	return WlayerSubbranchAveraged
			
def applyDBweights(AprevLayer, Wlayer):	
	if(useDependentSubbranches):
		ZsubbranchList = []
		for subbranchIndex in range(numberOfDependentSubbranches):	
			WlayerSubbranchAveraged = equaliseWlayerSubbranchValues(subbranchIndex, Wlayer, True)
		
			WlayerSubbranchAveraged = tf.reshape(WlayerSubbranchAveraged, [WlayerSubbranchAveraged.shape[1], WlayerSubbranchAveraged.shape[2]*WlayerSubbranchAveraged.shape[0]])
			Zsubbranch = tf.matmul(AprevLayer, WlayerSubbranchAveraged)
			Zsubbranch = tf.reshape(Zsubbranch, [Zsubbranch.shape[0], numberOfIndependentDendriticBranches, Zsubbranch.shape[1]//numberOfIndependentDendriticBranches])
			
			ZsubbranchList.append(Zsubbranch)
			
		Z = tf.stack(ZsubbranchList, axis=0)
		Z = tf.reduce_mean(Z, axis=0)
	else:
		Wlayer = tf.reshape(Wlayer, [Wlayer.shape[1], Wlayer.shape[2]*Wlayer.shape[0]])
		Z = tf.matmul(AprevLayer, Wlayer)
		Z = tf.reshape(Z, [Z.shape[0], numberOfIndependentDendriticBranches, Z.shape[1]//numberOfIndependentDendriticBranches])

	#take the highest output dendritic branch for each output neuron - WTA winner takes all
	Z = tf.math.reduce_max(Z, axis=1)
	
	if(normaliseActivationSparsity):
		Z = Z-weightStddev #reduce activation (since taking max value across independent segments will tend to be negative)
			
	return Z
					
def neuralNetworkPropagationANN(x, networkIndex=1, l=None):
	#print("numberOfLayers", numberOfLayers)

	if(l == None):
		maxLayer = numberOfLayers
	else:
		maxLayer = l
			
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	
	for l1 in range(1, maxLayer+1):
		if(supportSkipLayers):
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(0, l1):
				Wlayer = W[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "W")]
				AprevLayer = Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")]
				Z = tf.add(Z, applyDBweights(AprevLayer, Wlayer))
			Z = tf.add(Z, B[generateParameterNameNetwork(networkIndex, l1, "B")])
		else:
			Wlayer = W[generateParameterNameNetwork(networkIndex, l1, "W")]
			Z = applyDBweights(AprevLayer, Wlayer)
			Z = tf.add(Z, B[generateParameterNameNetwork(networkIndex, l1, "B")])
		A = activationFunction(Z)
		
		if(debugNormaliseActivationSparsity):
			#verify renormalised activation sparsity ~50%
			print("A = ", A) 

		#print("l1 = " + str(l1))		
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l1, "W")] )
		
		if(debugOnlyTrainFinalLayer):
			if(l1 < numberOfLayers):
				A = tf.stop_gradient(A)

		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
						
		AprevLayer = A

	if(maxLayer == numberOfLayers):
		return tf.nn.softmax(Z)
	else:
		return A

def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	

