import numpy as np
import math
import random
from mnist_loader import load_data, form_vector
import matplotlib.pyplot as plt

class NeuralNet:

	def __init__(self, num_units):
		self.num_layers = len(num_units)
		self.num_units = num_units
		self.weights = [(np.random.randn(x+1,y)).astype(float) for x,y in zip(num_units[:-1], num_units[1:])]     #(x+1) to accomodate the bias unit,    			weight_ dimensions =[from_layer, to_layer]
		

	def FeedForward(self, X, dropout = 0.0, batch_norm = 0):    #X is a numpy column		
		self.activations = [X]         #store activation values, first column being equal to the input values
		self.z = [X]		       #store z values for each layer, to be used in backpropagation
		for i in range(len(self.weights)):
			drop_coeff = np.ones(self.activations[-1].shape)			
			if i>0 and dropout>0.0:
				mute = [random.randint(0,len(self.z[-1])-1) for p in range(0,int(dropout*len(self.z[-1])))]				
				for j in mute:
					drop_coeff[j] = 0.0				
			drop_coeff = np.vstack((np.ones(1), drop_coeff))	
			self.z[-1] = np.vstack((np.ones(1), self.z[-1]))  #stack a [0] on top of each column for accomodating bias units
			self.activations[-1] = np.vstack((np.ones(1), self.activations[-1])) #stack a [0] on top of each column for accomodating bias units
			self.z.append(np.dot(self.weights[i].T, (self.activations[-1]*drop_coeff)).astype(float))			
			self.activations.append(self.sigmoid(np.dot(self.weights[i].T, (self.activations[-1]*drop_coeff))).astype(float))    
 			
	
	def sigmoid(self, x):         # sigmoid activation function
		return (1.0/(1.0 + np.exp(-x)))

	def sigmoid_derivative(self,x):  #sigmoid derivative function (for backpropagation)
		return self.sigmoid(x)*(1.0-self.sigmoid(x))

	def ReLU(self,x):
		x[x<0.0] = 0.0
		return x;
	
	def ReLU_derivative(self,x):
		x[x<=0.0] = 0.0
		x[x>0.0] = 1.0
		return x;

	def tanh(self, x):
		return 2.0*self.sigmoid(2.0*x) - 1.0 

	def tanh_derivative(self,x):
		return (1.0 - self.tanh(x)**2)

	def divide_and_learn(self, training_set, batch_size, eta, epochs, lmbda, num_iter, cost_function, test_data = None, L1_Reg = False, L2_Reg = False, dropout = 0.0, batch_norm = 0.0):
		all_test=[]              #Store the number of incorrect test_set predictions for all the epochs
		all_train=[]		 #Store the number of incorrect train_set predictions for all the epochs
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
			test_pred = self.evaluate_accuracy(test_data, 1)
			train_pred = self.evaluate_accuracy(self.train, 0)
			if test_data:
				 print "Epoch {0} Test data: {1} / {2} Training data: {3} / {4}".format(j, test_pred, num_test, train_pred, self.num_train)
			else:
				 print "Epoch {0} complete".format(j)
			all_train.append(float(100 - (float(train_pred)/float(self.num_train))*100))
			all_test.append(float(100 - (float(test_pred)/float(num_test))*100))
		fig1 = plt.figure(num_iter+15)
		plt.plot(np.arange(1, epochs+1), all_train)
		plt.axis([0, 100, 0, 100])
		plt.xlabel('Number of epochs', fontsize = 18)
		plt.ylabel('Training Error %', fontsize = 18)
		fig1.suptitle('Training Error')
		fig1.savefig(str(num_iter)+' train_2.png', dpi=600)


		fig2 = plt.figure(num_iter+30)
		plt.plot(np.arange(1, epochs+1, 1), all_test)
		plt.axis([0, 100, 0, 100])
		plt.xlabel('Number of epochs', fontsize = 18)
		plt.ylabel('Test Error %', fontsize = 18)
		fig2.suptitle('Test Error')
		fig2.savefig(str(num_iter)+' test_2.png', dpi=600)
		

	def backprop(self, x, label, dropout, cost_function, batch_norm):
		self.FeedForward(x, dropout, batch_norm)
		if cost_function == 1:                         #MSE cost function
			delta_L = self.cost_derivative_MSE(label)
		else:                                          #Cross Entropy cost function
			delta_L = self.cost_derivative_CEntropy(label)	
		del_weight_L = np.dot(self.activations[-2], delta_L.T)
		del_weights = [del_weight_L];
		delta = [delta_L];
		for i in range(len(self.weights) - 1):					
			delta.insert(0,(np.dot(self.weights[-i-1][1:,:], delta[-1-i])*self.sigmoid_derivative(self.z[-i-2][1:,:])))			
		for j in range(len(self.weights) - 1):
			del_weights.insert(0, np.dot(self.activations[-j-3], delta[-j-2].T))
		return del_weights

	def learn_NN(self, X, eta, batch_size, lmbda, cost_function, L1_Reg = False, L2_Reg = False, dropout = 0.0, batch_norm = 0):
		add_weight = [np.zeros(i.shape) for i in self.weights]
		reg_weight = [np.zeros(i.shape) for i in self.weights]
		if L2_Reg:
			reg_weight = [(1-float((lmbda*eta)/self.num_train))*k for k in self.weights]
		elif L1_Reg:
			temp = [np.sign(k) for k in self.weights]
			reg_weight = [l -(float((lmbda*eta)/self.num_train))*k for l,k in zip(self.weights,temp)]
		else:
			reg_weight = [k for k in self.weights]
		for j in range(len(self.weights)):						#revert back the regularization changes from the bias layers
			reg_weight[j][0,:] = self.weights[j][0,:]		

		batch_X = np.concatenate([arr for (arr,l) in X])
		if batch_norm:                                      #apply batch normalization only if batch_norm = 1 
			mean = np.mean(batch_X, axis = 0)
			var = np.var(batch_X, axis = 0)	
			for x,_label in X:
				x = (x - mu) / np.sqrt(var + 1e-8)				
				del_weight = self.backprop(x, _label, dropout, cost_function, batch_norm)
				a = [i.astype(float) for i in del_weight]
				del_weight = a
				add_weight = [m + eta*n for m,n in zip(add_weight, del_weight)]
		else:
			for x,_label in X:			
				del_weight = self.backprop(x, _label, dropout, cost_function, batch_norm)
				a = [i.astype(float) for i in del_weight]
				del_weight = a
				add_weight = [m + eta*n for m,n in zip(add_weight, del_weight)]
			
		self.weights = [f - (g/float(batch_size)) for f,g in zip(reg_weight, add_weight)]

	def cost_function_MSE(self, label):
		cost = np.zeros(np.shape(self.activations[-1])).astype(float)
		cost = (self.activations[-1] - label)**2
		return np.sum(cost)

	def cost_derivative_MSE(self, label):
		cost_der = np.zeros(np.shape(self.activations[-1])).astype(float)
		cost_der = self.activations[-1] - label
		cost_der = cost_der*self.sigmoid_derivative(self.z[-1])
		return cost_der

	def cost_function_CEntropy(self, label):
		cost = np.zeros(np.shape(self.activations[-1]))
		inv_activations = np.ones(np.shape(self.activations[-1])) - self.activations[-1]
		inv_label = np.ones(np.shape(y)) - y
		cost = label*np.log(self.activations[-1]) + inv_label*np.log(inv_activations)
		return -1*np.sum(cost)

	def cost_derivative_CEntropy(self, label):
		cost_der = np.zeros(np.shape(self.activations[-1])).astype(float)
		cost_der = self.activations[-1] - label
		return cost_der

	def evaluate_accuracy(self, data, convert):
		results = []
		for x,y in data:
			self.FeedForward(x, dropout = 0.0, batch_norm = 0)
			if convert == 1:
				results.append((np.argmax(self.activations[-1]),y)) # test_set has different representation, with the label being a number between 0-9
			else: 
				results.append((np.argmax(self.activations[-1]), np.argmax(y)))#training_set has label of a zeros vector except index = number being 1
        	return sum(int(np.array_equal(p,q)) for (p, q) in results)

	def evaluate_cost(self, data, convert, cost_function, reg, lmbda):
		overall_cost = 0.0
		if cost_function == 1:                            # if MSE cost requested
			if convert == 1:                          # test set, convert labels for cost calculation
				for a,b in data:
					self.FeedForward(a, dropout = 0.0, batch_norm = 0)	
					nx = form_vector(b)	
					overall_cost += self.cost_function_MSE(nx)

			else:
				for a,b in data:
					self.FeedForward(a, dropout = 0.0, batch_norm = 0)	
					overall_cost += self.cost_function_MSE(b)
		else:                                             # if CEntropy cost requested
			if convert == 1:
				for a,b in data:
					self.FeedForward(a, dropout = 0.0, batch_norm = 0)
					nx = form_vector(b)	
					overall_cost += self.cost_function_CEntropy(nx)

			else:
				for a,b in data:
					self.FeedForward(a, dropout = 0.0, batch_norm = 0)	
					overall_cost += self.cost_function_CEntropy(b)
		overall_cost = overall_cost/len(data)

		if reg==1:                                       # if L1 regularized cost is requested	
			overall_cost += (lmbda/len(data))*sum(np.sign(w[1:,:]) for w in self.weights)
		elif reg==2:                                     # if L2 regularized cost is requested
			overall_cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w[1:,:])**2 for w in self.weights)
		return overall_cost
				
	

training_data, test_data = load_data()
net = NeuralNet([784, 30, 6])
net.divide_and_learn(training_data, 10, 0.001, 100, 1, 1, 1, test_data = test_data, L1_Reg = False, L2_Reg = False, dropout = 0.0, batch_norm = 0)
#1. 0.01 lr
#2. Sigmoid 1000 lmbda

