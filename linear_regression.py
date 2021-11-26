import pandas as pd
import numpy as np
import os


def get_train_data():
    df = pd.read_csv("data.csv") # datafram (read data in file)

    X = np.array(df['km'].values)
    X_old = X
    X = scaling(X, max(X))
    X = np.c_[np.ones(X.shape[0]), X]

    y = np.array(df['price'].values)
    y_old = y
    y = scaling(y, max(y))

    return X, X_old, y, y_old

def scaling(data, scale):
    return data / scale

def	linear_regression_thetas(thetas, X, y, learning_rate, m):
	while 1:
		old_thetas = thetas
		# https://towardsdatascience.com/an-overview-of-the-gradient-descent-algorithm-8645c9e4de1e#:~:text=The%20general%20formula%20for%20getting%20consecutive%20theta%20value
		thetas = thetas - learning_rate * (1 / m) * (X.T @ ((X @ thetas) - y))
		if np.array_equal(thetas, old_thetas):
			break
	return thetas

def	calcul_thetas(X, X_old, y, y_old, learning_rate):
	m = float(len(X))
	thetas = np.array([0,0])
	thetas = linear_regression_thetas(thetas, X, y, learning_rate, m)
	thetas[0] = thetas[0] * max(y_old)
	thetas[1] = thetas[1] * (max(y_old) / max(X_old))
	return thetas

def	thetas_to_csv(thetas):

	mode = 'r+' if os.path.exists("thetas.csv") else 'w'
	with open("thetas.csv", mode) as fd:
		data = "theta0,theta1\n" + str(thetas[0]) + ',' + str(thetas[1]) + '\n'
		fd.write(data)

if __name__ == "__main__":
		X, X_old, y, y_old = get_train_data()
		thetas = calcul_thetas(X, X_old, y, y_old, 1)
		thetas_to_csv(thetas)
		print("\033[32;3mThetas are in thetas.csv \033[0m")