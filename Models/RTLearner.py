import math

import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  

class RTLearner(object):
	"""
	This is a Random Tree Learner. .

	:param verbose: If “verbose” is True, your code can print out information for debugging.
		If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
	:type verbose: bool
	"""
	def __init__(self, leaf_size, verbose=False):
		"""
		Constructor method
		"""

		self.leaf_size = leaf_size
		self.verbose = verbose
		self.tree = None

	def add_evidence(self, data_x, data_y):
		"""  		  	   		  		 		  		  		    	 		 		   		 		  
		Add training data to learner  		  	   		  		 		  		  		    	 		 		   		 		  
																							  
		:param data_x: A set of feature values used to train the learner  		  	   		  		 		  		  		    	 		 		   		 		  
		:type data_x: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
		:param data_y: The value we are attempting to predict given the X data  		  	   		  		 		  		  		    	 		 		   		 		  
		:type data_y: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
		"""
		self.tree = self.build_tree(data_x, data_y)
		return self.tree

	def build_tree(self, x_train, y_train):
		# Base Case -- Y is smaller than leaf size or all Y is the same
		if y_train.size <= self.leaf_size or np.unique(y_train).shape[0] == 1: #.shape[0]
			return np.array([["leaf", np.mean(y_train), None, None]])

		# RANDOM
		factor_index = np.random.randint(0, x_train.shape[1])

		split_val = np.median(x_train[:, factor_index])

		# Calculate the right and left training data based on above split_val
		left_train_bool = x_train[:, factor_index] <= split_val
		right_train_bool = x_train[:, factor_index] > split_val

		# Prevents infinite looping
		if np.all(left_train_bool) or np.all(right_train_bool):
			return np.array([["leaf", np.mean(y_train), None, None]])

		larry = self.build_tree(x_train[left_train_bool], y_train[left_train_bool])
		roger = self.build_tree(x_train[right_train_bool], y_train[right_train_bool])

		root = np.array([[factor_index, split_val, 1, larry.shape[0]]]) # Change to +1

		return np.vstack((root, larry, roger))




	def query(self, points):
		"""  		  	   		  		 		  		  		    	 		 		   		 		  
		Estimate a set of test points given the model we built.  		  	   		  		 		  		  		    	 		 		   		 		  
																							  
		:param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 		  		  		    	 		 		   		 		  
		:type points: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
		:return: The predicted result of the input data according to the trained model  		  	   		  		 		  		  		    	 		 		   		 		  
		:rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
		"""
		i = 0
		df = np.empty((points.shape[0]), float)
		for point in points:
			df[i] = self.query_one(point)
			i += 1
		return df

	def query_one(self, x):
		i = 0
		node = self.tree[i]
		while node[0] != 'leaf':
			factor = int(node[0])
			if x[factor] <= node[1]:
				i += int(node[2])
				node = self.tree[i]
			else:
				i += int(node[3])
				node = self.tree[i]

		return node[1]