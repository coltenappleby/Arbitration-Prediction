import numpy as np
import pandas as pd
import BagLearner as bl
import RTLearner as rt
import math
from matplotlib import pyplot as plt


# # Import the model we are using
# from sklearn.ensemble import RandomForestRegressor

if "__main__" == __name__:
	# load in the data
	filename = './data/arb-predictor-data_bat.csv'
	data = pd.read_csv(filename)

	data.dropna(subset=['G', 'Year Salary'], inplace=True)
	data = data.loc[~((data['Season'] < 2024) & (data['Next Year Salary'].isna()))]

	identifier = data['Player']
	data['Player'] = data.pop('Player')
	data['playerid'] = data.pop('playerid')

	train_x = data[data['Season'] < 2023].drop(['Next Year Salary'], axis=1).values
	train_x = train_x[:, :-2]
	train_y = data[data['Season'] < 2023]['Next Year Salary'].values

	test_x = data[data['Season'] == 2023].drop(['Next Year Salary'], axis=1).values
	test_y = data[data['Season'] == 2023]['Next Year Salary'].values
	test_names = data[data['Season'] == 2023]['Player'].values

	test2024_x = data[data['Season'] == 2024].drop(['Next Year Salary'], axis=1).values
	test2024_y = data[data['Season'] == 2024]['Next Year Salary'].values
	test2024_names = data[data['Season'] == 2024]['Player'].values

	# train the model
	leaf_size = 5
	bags = 10

	### BAG LEARNER ####
	bagTrainRMSE = np.empty(50, float)
	bagTestRMSE = np.empty(50, float)

	salary_2023 = np.empty((50, test_y.shape[0]), float)
	salary_2024 = np.empty((50, test2024_y.shape[0]), float)

	for i in range(50):
		learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False,
								verbose=False)
		learner.add_evidence(train_x, train_y)  # train it
		# Train
		pred_y = learner.query(train_x)
		TrainRmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  # TestRMSE
		bagTrainRMSE[i] = TrainRmse
		# Test
		pred_y = learner.query(test_x)
		salary_2023[i] = pred_y
		TestRmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  # TestRMSE
		bagTestRMSE[i] = TestRmse
		salary_2024[i] = learner.query(test2024_x)


	#### Save the Estimates ####
	salary_2023_mean = np.mean(salary_2023, axis=0)
	salary_2023_estimate = np.column_stack((test_names, test_y, salary_2023_mean))
	df = pd.DataFrame(salary_2023_estimate)
	df.to_csv('./data/salary_2023_estimate.csv')

	salary_2024_mean = np.mean(salary_2024, axis=0)
	salary_2024_estimate = np.column_stack((test2024_names, salary_2024_mean))
	df = pd.DataFrame(salary_2024_estimate)
	df.to_csv('./data/salary_2024_estimate.csv')






	#
	#
	# print(f"Test RMSE: {bagTestRMSE}")
	# print(f"Train RMSE: {bagTrainRMSE}")
	# # add code to plot here
	# fig2, ax2 = plt.subplots()
	# ax2.plot(bagTestRMSE)
	# ax2.plot(bagTrainRMSE)
	# ax2.axis(xmin=1, xmax=50, ymin=0, ymax=.015)
	# ax2.grid()
	# ax2.legend(['Test RMSE', 'Train RMSE'])
	# ax2.set_xlabel("Number of Leafs")
	# ax2.set_ylabel("RMSE")
	# ax2.set_title("Test and Train RMSE on Size of Leaf")
	#
	# fig2.savefig("./Figure1.png")
	# # fig2.show()
	# plt.close(fig2)
