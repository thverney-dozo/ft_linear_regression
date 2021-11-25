import pandas as pd
import numpy as np

def get_train_data():
    df = pd.read_csv("data.csv") # datafram (read data in file)
    y = np.array(df.price.values)
    y_old = y
    y = scaling(y, max(y))
    # df.drop("price", axis=1, inplace=True)
	X = np.array(df.km.values)
    X_old = X
    X = scaling(X, max(X))
    print(len(X))
    return X, X_old, y, y_old

def scaling(data, scale):
    return data / scale

def	calcul_thetas(X, Y, X_old, Y_old, learning_rate):
        thetas = np.array([0,0])

if __name__ == "__main__":
    X, X_old, y, y_old = get_train_data()

    print(len(X_train))

