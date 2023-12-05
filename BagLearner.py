import math

import numpy as np
import DTLearner as dt
import RTLearner as rt

class BagLearner(object):
	"""
	This is the BAGLEARNER
	:param learner: a learner
	:type learner: learner class

	:param verbose: If “verbose” is True, your code can print out information for debugging.
		If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
	:type verbose: bool
	"""
	def __init__(self, learner, kwargs={}, bags = 20 , boost = False, verbose = False):
		"""
		Constructor method
		"""
		self.learnerModel = learner
		self.kwargs = kwargs
		self.boost = boost
		self.verbose = verbose
		self.learners = np.empty(bags, learner)

		# create and train learners
		for i in range(bags):
			# create a learner and train it
			self.learners[i] = learner(**kwargs) # create a learner model

	def get_data(self, x_train, y_train):
		indices = np.random.choice(len(x_train), int(y_train.shape[0]*.6), replace=False)
		return x_train[indices], y_train[indices]

	def add_evidence(self, x_train, y_train):
		for learner in self.learners:
			x_train, y_train = self.get_data(x_train, y_train)
			learner.add_evidence(x_train, y_train)

	def query(self, x_test):
		"""
		:param x_test: npArray.
		returns y_pred -- mean of all y_preds for each learner type: [numpy.ndarray]
		"""
		y_preds = np.empty((x_test.shape[0]), float)
		for learner in self.learners:
			y_preds += learner.query(x_test)

		return y_preds/len(self.learners)


if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
	pass