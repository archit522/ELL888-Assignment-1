
import numpy as np
import math
import random
from mnist_loader import load_data, load_data_wrapper, vectorized_result

class NeuralNet:

	def __init__(self, num_units):
		self.num_layers = len(num_units)
		self.num_units = num_units
		self.weights = [(np.random.randn(x+1,y)/np.sqrt(x)).astype(float) for x,y in zip(num_units[:-1], num_units[1:])]     #(x+1) to accomodate the bias unit, dimensions =[from_layer, to_layer]
		self.learn_rate = 0.1
		#print self.weights
		#print'$$$$$'

	def FeedForward(self, X, dropout = 0.0):    #X is a numpy column
		self.activations = [X]         #store activation values, first column being equal to the input values
		#self.activations_no_bias = [X]		
		#self.z_no_bias = [X]
		self.z = [X]		       #store z values for each layer, to be used in backpropagation
		for i in range(len(self.weights)):
			drop_coeff = np.ones(self.activations[-1].shape)			
			if i>0 and dropout>0.0:
				#print "Len z:" len(self.z[-1])
				mute = [random.randint(0,len(self.z[-1])-1) for p in range(0,int(dropout*len(self.z[-1])))]				
				for j in mute:
					drop_coeff[j] = 0.0				
			drop_coeff = np.vstack((np.ones(1), drop_coeff))	
			self.z[-1] = np.vstack((np.ones(1), self.z[-1]))
			self.activations[-1] = np.vstack((np.ones(1), self.activations[-1]))
			#print "Shape_weights:", np.shape(self.weights[i]), "Activations:", np.shape(np.vstack((np.ones(1), self.activations[-1])))
			self.activations.append(self.sigmoid(np.dot(self.weights[i].T, (self.activations[-1]*drop_coeff))).astype(float))     #stack a [0] on top of each column for accomodating bias units			
			#self.activations.append(self.sigmoid(np.dot(self.weights[i].T, self.activations[-1])).astype(float))
			#self.activations_no_bias.append(sigmoid(np.dot(self.weights[i], np.vstack((np.zeros(1), self.activations_no_bias[-1])))))			
			#self.z_no_bias.append(np.dot(self.weights[i], np.vstack((np.ones(1), self.z_no_bias[-1]))))
			self.z.append(np.dot(self.weights[i].T, (self.z[-1]*drop_coeff)).astype(float))
			#print "Z_size:", np.shape(self.z[i]), "Activations_size:", np.shape(self.activations[i])
		#print self.activations[1]
		#print self.activations[2]	
	
	def sigmoid(self, x):         # sigmoid activation function
		return (1.0/(1.0 + np.exp(-x)))

	def sigmoid_derivative(self,x):  #sigmoid derivative function (for backpropagation)
		return self.sigmoid(x)*(1.0-self.sigmoid(x))

	def ReLU(self,x):
		return np.array([np.maximum(i,0.0) for i in np.nditer(x)]).reshape(a.shape)
	
	def ReLU_derivative(self,x):
		return np.array([1.0 if i>0 else 0.0 for i in np.nditer(x)]).reshape(a.shape)

	def tanh(self, x):
		return 2.0*self.sigmoid(2.0*x) - 1.0 

	def tanh_derivative(self,x):
		return (1.0 - self.tanh(x)**2)

	def divide_and_learn(self, training_set, batch_size, eta, epochs, lmbda, test_data = None, L1_Reg = False, L2_Reg = False, dropout = 0.0):
		num_test = len(test_data)
		self.num_train = len(training_set)
		self.train = training_set		
		for j in xrange(epochs):			
			random.shuffle(training_set)
			batches = []
			for i in np.arange(0, len(training_set), batch_size):
				batches.append(training_set[i:i+batch_size])
			for batch in batches:
				self.learn_NN(batch, eta, batch_size, lmbda, L1_Reg, L2_Reg, dropout)
			if test_data:
				 print "Epoch {0} Test data: {1} / {2} Training data: {3} / {4}".format(j, self.evaluate(test_data, 1), num_test, self.evaluate(self.train, 0), self.num_train) 
			else:
				 print "Epoch {0} complete".format(j)


	def backprop(self, label):
		#last_layer_cost = self.cost_derivative_MSE_noReg(label)
		#delta_L = last_layer_cost*self.sigmoid_derivative(self.z[-1])
		delta_L = self.cost_derivative_CEntropy_noReg(label)
		del_weight_L = np.dot(self.activations[-2], delta_L.T)
		del_weights = [del_weight_L];
		delta = [delta_L];
		for i in range(len(self.weights) - 1):	
			#print "Shape_delta:", np.shape(np.dot(self.weights[-i-1], delta[-1-i])), "z:", np.shape(self.sigmoid_derivative(self.z[-i-2]))
			delta.insert(0,(np.dot(self.weights[-i-1][1:,:], delta[-1-i])*self.sigmoid_derivative(self.z[-i-2][1:,:])))
		#print "Length of delta", len(delta)
		for j in range(len(self.weights) - 1):
			del_weights.insert(0, np.dot(self.activations[-j-3], delta[-j-2].T))
			#print del_weights[0]
		return del_weights

	def learn_NN(self, X, eta, batch_size, lmbda, L1_Reg = False, L2_Reg = False, dropout = 0.0):
		add_weight = [np.zeros(i.shape) for i in self.weights]
		reg_weight = [np.zeros(i.shape) for i in self.weights]
		for x,_label in X:
			self.FeedForward(x, dropout)
			del_weight = self.backprop(_label)
			for a in del_weight:
				a = a.astype(float)
			if L2_Reg:
				reg_weight = [(1-(lmbda*eta)/self.num_train)*k for k in self.weights]
			elif L1_Reg:
				temp = [np.sign(k) for k in self.weights]
				reg_weight = [(1 -(lmbda*eta)/self.num_train)*k for k in temp]
			else:
				reg_weight = [k for k in self.weights]
			for j in range(len(self.weights)):										#revert back the regularization changes from the bias layers
				reg_weight[j][0,:] = self.weights[j][0,:]
			add_weight = [m + eta*n for m,n in zip(add_weight, del_weight)]
		self.weights = [f - (g/batch_size) for f,g in zip(reg_weight, add_weight)]

	def cost_function_MSE_noReg(self, label):
		cost = np.zeros(np.shape(self.activations[-1])).astype(float)
		cost = (self.activations[-1] - label)**2
		return np.sum(cost)

	def cost_derivative_MSE_noReg(self, label):
		cost_der = np.zeros(np.shape(self.activations[-1])).astype(float)
		cost_der = self.activations[-1] - label
		return cost_der

	def cost_function_CEntropy_noReg(self, label):
		cost = np.zeros(np.shape(self.activations[-1]))
		inv_activations = np.ones(np.shape(self.activations[-1])) - self.activations[-1]
		inv_label = np.ones(np.shape(y)) - y
		cost = label*np.log(self.activations[-1]) + inv_label*np.log(inv_activations)
		return -1*np.sum(cost)

	def cost_derivative_CEntropy_noReg(self, label):
		cost_der = np.zeros(np.shape(self.activations[-1])).astype(float)
		cost_der = self.activations[-1] - label
		return cost_der

	def cost_function_MSE_L2Reg(self, label, lmbda):
		cost = np.zeros(np.shape(self.activations[-1])).astype(float)
		cost = self.activations[-1] - label
		reg_term = 0.0	
		for i in self.weights:
			reg_term = reg_term + np.sum(i**2) 
		cost = cost + reg_term*lmbda/2
		cost = cost/2
		return np.sum(cost)

	def cost_function_CEntropy_L2Reg(self, label, lmbda):
		cost = np.zeros(np.shape(self.activations[-1]))
		inv_activations = np.ones(np.shape(self.activations[-1])) - self.activations[-1]
		inv_label = np.ones(np.shape(y)) - y
		cost = label*np.log(self.activations[-1]) + inv_label*np.log(inv_activations)
		reg_term = 0.0	
		for i in self.weights:
			reg_term = reg_term + np.sum(i**2)
		cost = cost - reg_term*lmbda/2
		return -1*np.sum(cost)

	def evaluate(self, data, convert):
		results = []
		for x,y in data:
			self.FeedForward(x, dropout = 0.0)
			if convert == 1:
				results.append((np.argmax(self.activations[-1]),y))                       # test_set has different representation, with the label being a number betwenn 0-9
			else: 
				results.append((np.argmax(self.activations[-1]), np.argmax(y)))		  # training_set has labels of a 1-d zeros vector except the index = number being 1
        	return sum(int(np.array_equal(x,y)) for (x, y) in results)
	


training_set, test_set = load_data_wrapper()
net = NeuralNet([784,30,10])
a1 = np.array([1, 0, -1])
z1 = net.sigmoid_derivative(a1)
#print z1
net.divide_and_learn(training_set, 10, 0.1, 30, 0.5, test_data = test_set, L1_Reg = False, L2_Reg = False, dropout = 0.0)
