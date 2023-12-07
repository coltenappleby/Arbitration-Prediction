import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from Models import BagLearner as bl, RTLearner as rt
import math
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
def RandomForest():
	# load in the data
	filename = 'data/arb-predictor-data_bat_step1.csv'
	data = pd.read_csv(filename)

	data.dropna(subset=['G', 'Year Salary'], inplace=True)
	data = data.loc[~((data['Season'] < 2024) & (data['Next Year Salary'].isna()))]

	identifier = data['Player']
	data['Player'] = data.pop('Player')
	data['playerid'] = data.pop('playerid')
	data['Season'] = data.pop('Season')

	x_train = data[data['Season'] < 2023].drop(['Next Year Salary'], axis=1).values
	x_train = x_train[:, :-3] # remove Player, playerid and Season
	y_train = data[data['Season'] < 2023]['Next Year Salary'].values

	x_test = data[data['Season'] == 2023].drop(['Next Year Salary'], axis=1).values
	y_test = data[data['Season'] == 2023]['Next Year Salary'].values
	test_names = data[data['Season'] == 2023]['Player'].values

	x_2024 = data[data['Season'] == 2024].drop(['Next Year Salary'], axis=1).values
	y_2024 = data[data['Season'] == 2024]['Next Year Salary'].values
	names_2024 = data[data['Season'] == 2024]['Player'].values

	# train the model
	leaf_size = 5
	bags = 20

	#### BAG LEARNER ####
	bagTrainRMSE = np.empty(50, float)
	bagTestRMSE = np.empty(50, float)

	salary_2023 = np.empty((50, y_test.shape[0]), float)
	salary_2024 = np.empty((50, y_2024.shape[0]), float)

	for i in range(50):
		learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False,
								verbose=False)
		learner.add_evidence(x_train, y_train)  # train it
		# Training Data
		pred_y = learner.query(x_train)
		TrainRmse = math.sqrt(((y_train - pred_y) ** 2).sum() / y_train.shape[0])  # TestRMSE
		bagTrainRMSE[i] = TrainRmse
		# Test
		pred_y = learner.query(x_test)
		salary_2023[i] = pred_y
		TestRmse = math.sqrt(((y_test - pred_y) ** 2).sum() / y_test.shape[0])  # TestRMSE
		bagTestRMSE[i] = TestRmse
		salary_2024[i] = learner.query(x_2024)

	#### Save the Estimates ####
	pred_y = np.mean(salary_2023, axis=0)
	salary_2023_estimate = np.column_stack((test_names, y_test, pred_y))
	df = pd.DataFrame(salary_2023_estimate)
	df.to_csv('./data/salary_2023_estimate.csv')

	avg_diff_2023 = np.mean(np.abs(y_test - pred_y))
	print(f"Average Difference 2023: {avg_diff_2023}")

	salary_2024_mean = np.mean(salary_2024, axis=0)
	salary_2024_estimate = np.column_stack((names_2024, salary_2024_mean))
	df = pd.DataFrame(salary_2024_estimate)
	df.to_csv('./data/salary_2024_estimate.csv')


	print(f"Test RMSE: {np.mean(bagTestRMSE)}")
	print(f"Train RMSE: {np.mean(bagTrainRMSE)}")

	residuals = y_test - pred_y

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

	plt.show()
	# plt.show()
	print(np.mean(y_test))
	print(y_test)




def LinearRegressionModel():
	filename = 'data/arb-predictor-data_bat_step1.csv'
	data = pd.read_csv(filename)

	data.dropna(subset=['G', 'Year Salary'], inplace=True)
	data = data.loc[~((data['Season'] < 2024) & (data['Next Year Salary'].isna()))]

	identifier = data['Player']
	data['Player'] = data.pop('Player')
	data['playerid'] = data.pop('playerid')
	data['Season'] = data.pop('Season')

	x_train = data[data['Season'] < 2023].drop(['Next Year Salary'], axis=1).values
	x_train = x_train[:, :-3] # remove Player, playerid and Season
	y_train = data[data['Season'] < 2023]['Next Year Salary'].values

	x_test = data[data['Season'] == 2023].drop(['Next Year Salary'], axis=1).values
	x_test = x_test[:, :-3] # remove Player, playerid and Season
	y_test = data[data['Season'] == 2023]['Next Year Salary'].values
	test_names = data[data['Season'] == 2023]['Player'].values

	# Create and fit the model
	model = LinearRegression()
	model.fit(x_train, y_train)

	# Make predictions on the test set
	y_pred = model.predict(x_test)

	# Evaluate the model
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	#### Save the Estimates ####
	# salary_2023_mean = np.mean(y_pred, axis=0)
	salary_2023_estimate = np.column_stack((test_names, y_test, y_pred))
	avg_diff_2023 = np.mean(np.abs(y_test - y_pred))
	print(f"Average Difference 2023: {avg_diff_2023}")


	# Print coefficients, intercept, and evaluation metrics
	print("Coefficients:", model.coef_)
	print("Intercept:", model.intercept_)
	print("Mean Squared Error:", mse)
	print("R-squared:", r2)

	print(data.columns)

if "__main__" == __name__:
	RandomForest()