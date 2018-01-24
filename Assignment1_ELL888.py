import numpy as np
import math
import random
from mnist_loader imoprt load_data, load_data_wrapper, vectorized_result

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
		return (1.0/(1.0 + np.exp(-x)))

	def sigmoid_derivative(self,x):  #sigmoid derivative function (for backpropagation)
		return (np.exp(-x)/((1 + math.exp(-x)) ** 2))

	def divide_and_learn(self, training_set = None, num_batches, batch_size, eta):
		training_set = load_data_wrapper()
		batches = []
		for i in np.arange(0, len(training_set), num_batches):
			batches.append(training_set[i:i+num_batches])
			for batch in batches:
				learn_NN(batch, eta, batch_size)


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

	def learn_NN(self, X, eta, batch_size):
		add_weight = 0;
		for x,_label in X:
			FeedForward(x)
			del_weight = self.backprop(_label)
			add_weight = add_weight + eta*del_weight
		self.weights = self.weights - (add_weight/batch_size)

	def cost_function_MSE_noReg(self, label):
		cost = np.zeros(np.shape(self.activations[-1]))
		cost = (self.activations[-1] - label)**2
		return np.sum(cost)

	def cost_derivative_MSE_noReg(self, label):
		cost_der = np.zeros(np.shape(self.activations[-1]))
		cost_der = self.activations[-1] - label
		return cost_der

	def cost_function_CEntropy_noReg(self, label):
		cost = np.zeros(np.shape(self.activations[-1]))
		inv_activations = np.ones(np.shape(self.activations[-1])) - self.activations[-1]
		inv_label = np.ones(np.shape(y)) - y
		cost = label*np.log(self.activations[-1]) + inv_label*np.log(inv_activations)
		return -1*np.sum(cost)

	def cost_derivative_CEntropy_noReg(self, label):
		cost_der = np.zeros(np.shape(self.activations[-1]))
