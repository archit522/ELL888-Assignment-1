import numpy as np
import math
import random

class NeuralNet:

	def init_NN(self, num_units):
		self.num_layers = len(num_units)
		self.num_units = num_units
		self.weights = [np.random.randn(x+1,y) for x,y in zip(sizes[:-1], sizes[1:])]     #(x+1) to accomodate the bias unit, dimensions =[from_layer, to_layer]
		self.learn_rate = 0.1

	def FeedForward(self, X):    #X is a numpy column
		self.activations = [X]         #store activation values, first column being equal to the input values
		self.z = [X]		       #store z values for each layer, to be used in backpropagation
		for i in range(len(self.weights)):
			self.activations.append(sigmoid(np.dot(self.weights[i], np.vstack(np.zeros(1), self.activations[-1]))))     #stack a [0] on top of each column for accomodating bias units
			self.z.append(np.dot(self.weights[i], np.vstack(np.zeros(1), self.z[-1])))


	def sigmoid(self, x):         # sigmoid activation function
		return (1.0/(1.0 + math.exp(-x)))

	def sigmoid_derivative(self,x):  #sigmoid derivative function (for backpropagation)
		return (math.exp(-x)/((1 + math.exp(-x)) ** 2))

	def backprop(self, labels):
		delta_L = del_cost(self.activations[-1], labels) * sigmoid_derivative(self.z[-1])
		del_weight_L = np.dot(delta_L, self.activations[-2].T)
		del_weights = [del_weight_L];
		delta = [delta_L];
		for i in range(len(self.weights)):
			delta.insert(0,(np.dot(self.weights[-i-1].T, delta[-1-i])*sigmoid_derivative(self.z[-i-2])))
		for j in range(len(self.weights) - 1):		
			del_weights.insert(0, (np.dot(delta[-i-2], self.activations[-i-3].T))) 
		
		return del_weights

	def learn_NN(self, X, labels, eta, batch_size):
		add_weight = 0;
		for x,_label in zip(X,labels):
			FeedForward(x)
			del_weight = self.backprop(_label)
			add_weight = add_weight + eta*del_weight
		self.weights = self.weights - (add_weight/batch_size)

	
			