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

	# data.drop(['Next Year Salary', 'Salary', 'SeasonStat'], axis=1, inplace=True)

	columns_to_keep = ['Player', 'Salary Diff', 'Salary', 'Next Year Salary', 'Year', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'AVG', 'TB']
	data = data.loc[:, columns_to_keep]

	columns_to_dro_cump = ['Player', 'Service', 'Year', 'Next Year Salary', 'Salary',
	 'Salary Diff', 'SeasonStat', 'Age', 'G', 'PA', 'AB', 'R', 'H', '2B',
	 '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS',
	 'GDP', 'HBP', 'SH', 'SF', 'IBB', 'WAR', 'Defense', 'Position', 'TB',
	 'cum_sum_G', 'cum_sum_PA', 'cum_sum_AB', 'cum_sum_R', 'cum_sum_H',
	 'cum_sum_2B', 'cum_sum_3B', 'cum_sum_HR', 'cum_sum_RBI', 'cum_sum_SB',
	 'cum_sum_CS', 'cum_sum_BB', 'cum_sum_SO', 'cum_sum_TB', 'cum_sum_GDP',
	 'cum_sum_HBP', 'cum_sum_SH', 'cum_sum_SF', 'cum_sum_IBB',
	 'cum_sum_WAR']

	# %%
	# dummies = pd.get_dummies(data['Position'], prefix='pos')
	# data = pd.concat([data, dummies], axis=1)
	# data.drop(['Position'], axis=1, inplace=True)

	# identifier = data['Player']
	data['Player'] = data.pop('Player')
	data['Year'] = data.pop('Year')
	data['Salary'] = data.pop('Salary')
	data['Next Year Salary'] = data.pop('Next Year Salary')
	# data['playerid'] = data.pop('playerid')
	# data['Season'] = data.pop('Season')

	stat_to_predict = 'Salary Diff'

	x_train = data[data['Year'] < 2023].drop([stat_to_predict], axis=1).values
	x_train = x_train[:, :-4]  # remove Player
	y_train = data[data['Year'] < 2023][stat_to_predict].values

	x_test = data[data['Year'] == 2023].drop([stat_to_predict], axis=1).values
	x_test = x_test[:, :-4]  # remove Player
	y_test = data[data['Year'] == 2023][stat_to_predict].values
	test_names = data[data['Year'] == 2023][['Player', 'Salary', 'Next Year Salary']].values

	x_2024 = data[data['Year'] == 2024].drop([stat_to_predict], axis=1).values
	x_2024 = x_2024[:, :-4]  # remove Player
	y_2024 = data[data['Year'] == 2024][stat_to_predict].values
	names_2024 = data[data['Year'] == 2024]['Player'].values

	return x_train, x_test, y_train, y_test, test_names, x_2024, y_2024, names_2024

def prediction_comparison(my_preds,filename):
	mlbtr_preds = pd.read_csv('data/mlbtr_2023_preds.csv')

	merged = pd.merge(my_preds, mlbtr_preds, on='Player', how='inner')

	merged['Salary2023 Pred'] = (merged['Salary2022'] + merged['PredictedInc']).astype('Int64')
	merged['Actual-ME'] = (merged['Actual2023Sal'] - merged['Salary2023 Pred']).astype('Int64')

	merged['TR-ME'] = (merged['TR Pred Salary'] - merged['Salary2023 Pred']).astype('Int64')
	merged['TR-A'] = (merged['Actual2023Sal'] - merged['TR Pred Salary']).astype('Int64')

	merged.drop(['Service'], axis=1, inplace=True)
	# merged['Predicted'] = merged['Predicted'].astype('Int64')
	# merged['Actual'] = merged['Actual'].astype('Int64')
	# merged['Delta'] = merged['Delta'].astype('Int64')

	merged.to_csv(filename)

	# print("Salary 2023")
	table = tabulate(merged, headers='keys', tablefmt='jira', showindex=False, intfmt=',', numalign="right")
	# print(table)
	# return mlbtr


def RandomForest(x_train, x_test, y_train, y_test, x_2024):


	# Build and Train the Model
	leaf_size = 5
	bags = 1000
	learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False,
							verbose=False)
	learner.add_evidence(x_train, y_train)  # train it

	# Training Data
	y_train_pred = learner.query(x_train)
	trainRmse = math.sqrt(((y_train - y_train_pred) ** 2).sum() / y_train.shape[0])  # TestRMSE

	# Testing Data
	y_pred = learner.query(x_test)
	testRmse = math.sqrt(((y_test - y_pred) ** 2).sum() / y_test.shape[0])  # TestRMSE

	# Prediction Data
	y_2024_pred = learner.query(x_2024)

	return y_pred, trainRmse, testRmse, y_2024_pred


def LinearRegressionModel(x_train, x_test, y_train, y_test, x_2024):

	x_train, x_test, y_train, y_test, test_names, x_2024, y_2024, names_2024 = preprocess()

	# Create and fit the model
	model = LinearRegression()
	model.fit(x_train, y_train)

	# Make predictions on the test set
	y_pred = model.predict(x_test)
	y_2024_pred = model.predict(x_2024)

	# Evaluate the model
	test_mse = mean_squared_error(y_test, y_pred)
	test_r2 = r2_score(y_test, y_pred)

	coefficients = model.coef_
	intercept = model.intercept_
	print("Parameters:")

	print(model.get_params())

	print("Coefficients:", coefficients)
	print("Intercept:", intercept)

	return y_pred, test_mse, test_r2, y_2024_pred

if "__main__" == __name__:

	x_train, x_test, y_train, y_test, test_names, x_2024, y_2024, names_2024 = preprocess()

	#### RANDOM FOREST ####
	# y_pred, trainRmse, testRmse, y_2024_pred = RandomForest(x_train, x_test, y_train, y_test, x_2024)
	#
	# #### Save the Estimates ####
	# y_pred = y_pred.astype(int)
	# y_test = y_test.astype(int)
	#
	# salary_2023_estimate = np.column_stack((test_names, y_test, y_pred))
	# df = pd.DataFrame(salary_2023_estimate, columns=['Player', 'Salary2022', 'Actual2023Sal', 'Actual2023Inc', 'PredictedInc'])
	# prediction_comparison(df, 'predictions/salary_2023_estimate_RF.csv')
	#
	# print("RANDOM FORREST")
	# # print(f"Average Difference 2023: {avg_diff_2023}")
	# print(f"Train RMSE: {trainRmse}")
	# print(f"Test RMSE: {testRmse}")

	#### LINEAR REGRESSION ####
	print("LINEAR REGRESSION")
	y_pred, test_mse, test_r2, y_2024_pred =  LinearRegressionModel(x_train, x_test, y_train, y_test, x_2024)

	#### Save the Estimates ####
	y_pred = y_pred.astype(int)
	y_test = y_test.astype(int)

	salary_2023_estimate = np.column_stack((test_names, y_test, y_pred))
	df = pd.DataFrame(salary_2023_estimate,
					  columns=['Player', 'Salary2022', 'Actual2023Sal', 'Actual2023Inc', 'PredictedInc'])
	prediction_comparison(df, 'predictions/salary_2023_estimate_LR.csv')


	# print(f"Average Difference 2023: {avg_diff_2023}")
	print(f"Train mse: {test_mse}")
	print(f"Test r^2: {test_r2}")