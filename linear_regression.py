import pandas as pd
import numpy as np
import os

def get_train_data():
	# datafram (read data in file)
	df = pd.read_csv("data.csv")
	# insert kms data into X
	X = np.array(df['km'].values)
	# temporary X to later unscale thetas with max(X)
	X_old = X
	# scaling to optimize operations such as np.square or @ (matrix multiplication)
	X = scaling(X, max(X))
	# np.c_ with np.ones and X to slice into a matrix
	X = np.c_[np.ones(X.shape[0]), X]
	# insert prices data into y
	y = np.array(df['price'].values)
	# temporary y to later unscale thetas with max(y) 
	y_old = y
	# same as X scaling
	y = scaling(y, max(y))

	return X, X_old, y, y_old

def scaling(data, scale):
	return data / scale

def cost(theta, X, y):
	# number of price values in y
	m = float(len(y))
	# https://miro.medium.com/max/804/1*p18Ryw6XXMR4u5q7WPX26w.png
	c = (1/2 * m) * np.sum(np.square((X.dot(theta)) - y))  
	return c

def	linear_regression_thetas(thetas, X, y, learning_rate, m):
	costs = []
	i = 0
	while 1:
		old_thetas = thetas
		# linear regression formula
		thetas = thetas - learning_rate * (1 / m) * (X.T @ ((X @ thetas) - y))
		# X.T.dot on two 2D arrays is the same as X.T @
		costs = np.append(costs,(cost(thetas, X, y)))
		i = i + 1
		if np.array_equal(thetas, old_thetas):
			break
	return costs, i, thetas

def	calcul_thetas(X, X_old, y, y_old, learning_rate):
	# number of km values in X
	m = float(len(X))
	thetas = np.array([0,0])
	costs, i, thetas = linear_regression_thetas(thetas, X, y, learning_rate, m)
	# unscaling
	thetas[0] = thetas[0] * max(y_old)
	thetas[1] = thetas[1] * (max(y_old) / max(X_old))
	return costs, i, thetas

def	thetas_to_csv(thetas):
	mode = 'r+' if os.path.exists("thetas.csv") else 'w'
	with open("thetas.csv", mode) as fd:
		data_thetas = "theta0,theta1\n" + str(thetas[0]) + ',' + str(thetas[1]) + '\n'
		fd.write(data_thetas)

def	costs_to_csv(costs, i):
	mode = 'r+' if os.path.exists("costs.csv") else 'w'
	with open("costs.csv", mode) as fd:
		data_costs = "iterations,costs\n"
		for index in range(i):
			if abs(costs[index] - costs[index + 1]) < 0.0001:
				break
			data_costs = data_costs + str(index) + ',' + str(costs[index]) + '\n'
			
		fd.write(data_costs)

if __name__ == "__main__":
		# https://towardsdatascience.com/an-overview-of-the-gradient-descent-algorithm-8645c9e4de1e
		X, X_old, y, y_old = get_train_data()
		costs, i, thetas = calcul_thetas(X, X_old, y, y_old, 1)

		thetas_to_csv(thetas)
		costs_to_csv(costs, i)
		print("\033[32;3mThetas are in thetas.csv \033[0m")
		print("\033[32;3mCosts are in costs.csv \033[0m")
		print("\033[32;3mMinimum cost is {}\033[0m".format(costs[i-1]))