import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from tabulate import tabulate

from Models import BagLearner as bl, RTLearner as rt
import math
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def preprocess():
	# load in the data
	filename = 'data/arb-data-with-stats_bats.csv'
	data = pd.read_csv(filename)

	# data.dropna(subset=['G', 'Year Salary'], inplace=True)
	# data = data.loc[~((data['Season'] < 2024) & (data['Next Year Salary'].isna()))]
	data.drop(['Salary Diff', 'Salary', 'SeasonStat'], axis=1, inplace=True)

	# %%
	dummies = pd.get_dummies(data['Position'], prefix='pos')
	data = pd.concat([data, dummies], axis=1)
	data.drop(['Position'], axis=1, inplace=True)

	# identifier = data['Player']
	data['Player'] = data.pop('Player')
	# data['playerid'] = data.pop('playerid')
	# data['Season'] = data.pop('Season')

	stat_to_predict = 'Next Year Salary'

	x_train = data[data['Year'] < 2023].drop([stat_to_predict], axis=1).values
	x_train = x_train[:, :-1]  # remove Player
	y_train = data[data['Year'] < 2023][stat_to_predict].values

	x_test = data[data['Year'] == 2023].drop([stat_to_predict], axis=1).values
	x_test = x_test[:, :-1]  # remove Player
	y_test = data[data['Year'] == 2023][stat_to_predict].values
	test_names = data[data['Year'] == 2023]['Player'].values

	x_2024 = data[data['Year'] == 2024].drop([stat_to_predict], axis=1).values
	y_2024 = data[data['Year'] == 2024][stat_to_predict].values
	names_2024 = data[data['Year'] == 2024]['Player'].values

	return x_train, x_test, y_train, y_test, test_names, x_2024, y_2024, names_2024

def RandomForest():

	x_train, x_test, y_train, y_test, test_names, x_2024, y_2024, names_2024 = preprocess()

	# train the model
	leaf_size = 5

	#### Multiple Bag Learners ####
	# bags = 20
	# runs = 50
	# bagTrainRMSE = np.empty(runs, float)
	# bagTestRMSE = np.empty(runs, float)
	# salary_2023 = np.empty((runs, y_test.shape[0]), float)
	# salary_2024 = np.empty((runs, y_2024.shape[0]), float)
	# for i in range(runs):
	# 	learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False,
	# 							verbose=False)
	# 	learner.add_evidence(x_train, y_train)  # train it
	# 	# Training Data
	# 	pred_y = learner.query(x_train)
	# 	TrainRmse = math.sqrt(((y_train - pred_y) ** 2).sum() / y_train.shape[0])  # TestRMSE
	# 	bagTrainRMSE[i] = TrainRmse
	# 	# Test
	# 	pred_y = learner.query(x_test)
	# 	salary_2023[i] = pred_y
	# 	TestRmse = math.sqrt(((y_test - pred_y) ** 2).sum() / y_test.shape[0])  # TestRMSE
	# 	bagTestRMSE[i] = TestRmse
	# 	salary_2024[i] = learner.query(x_2024)

	bags = 100
	learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False,
							verbose=False)

	learner.add_evidence(x_train, y_train)  # train it
	# Training Data
	y_pred = learner.query(x_train)
	trainRmse = math.sqrt(((y_train - y_pred) ** 2).sum() / y_train.shape[0])  # TestRMSE

	# Test
	y_pred = learner.query(x_test)
	testRmse = math.sqrt(((y_test - y_pred) ** 2).sum() / y_test.shape[0])  # TestRMSE

	salary_2024 = learner.query(x_2024)

	#### Save the Estimates ####
	# y_pred = np.mean(salary_2023, axis=0)
	y_pred = y_pred.astype(int)
	y_test = y_test.astype(int)
	delta = y_test - y_pred
	salary_2023_estimate = np.column_stack((test_names, y_test, y_pred, delta))
	df = pd.DataFrame(salary_2023_estimate, columns=['Player', 'Actual', 'Predicted', "Difference"])
	df.to_csv('./predictions/salary_2023_estimate.csv')

	avg_diff_2023 = np.mean(np.abs(y_test - y_pred))

	print("Salary 2023")
	table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False, intfmt=",")
	print(table)
	print(f"Average Difference 2023: {avg_diff_2023}")
	print(f"Train RMSE: {trainRmse}")
	print(f"Test RMSE: {testRmse}")

	# salary_2024_mean = np.mean(salary_2024, axis=0)
	salary_2024_estimate = np.column_stack((names_2024, salary_2024))
	# df = pd.DataFrame(salary_2024_estimate)
	# df.to_csv('./predictions/salary_2024_estimate.csv')



	residuals = y_test - y_pred

	# Create a histogram of residuals
	plt.hist(y_test, bins=20, color='blue', alpha=0.7)
	plt.title('Histogram of Actual Salaries')
	plt.xlabel('Values')
	plt.ylabel('Frequency')

	# Function to format y-axis labels in millions
	def millions_formatter(x, pos):
		return f'{x / 1e6:.1f}M'

	# Apply the formatter to the y-axis
	plt.gca().yaxis.set_major_formatter(FuncFormatter(millions_formatter))

	# plt.show()
	# plt.show()
	# print(np.mean(y_test))
	# print(y_test)


def LinearRegressionModel():

	x_train, x_test, y_train, y_test, test_names, x_2024, y_2024, names_2024 = preprocess()

	# Create and fit the model
	model = LinearRegression()
	model.fit(x_train, y_train)

	# Make predictions on the test set
	y_pred = model.predict(x_test)

	# Evaluate the model
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	#### Save the Estimates ####
	pred_y = y_pred.astype(int)
	y_test = y_test.astype(int)
	delta = y_test - pred_y
	salary_2023_estimate = np.column_stack((test_names, y_test, pred_y, delta))
	df = pd.DataFrame(salary_2023_estimate, columns=['Player', 'Actual', 'Predicted', "Difference"])
	print("Salary 2023")
	table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False, intfmt=",")
	print(table)
	# df.to_csv('./predictions/salary_2023_estimate.csv')
	avg_diff_2023 = np.mean(np.abs(y_test - y_pred))
	print(f"Average Difference 2023: {avg_diff_2023}")


	# Print coefficients, intercept, and evaluation metrics
	print("Coefficients:", model.coef_)
	print("Intercept:", model.intercept_)
	print("Mean Squared Error:", mse)
	print("R-squared:", r2)

	# print(data.columns)

if "__main__" == __name__:
	LinearRegressionModel()