import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
	description="Predict a car's price.")
parser.add_argument('kms', help="car's mileage")
parser.add_argument('--graph', '-g', action='store_true', help='show graph')
options = parser.parse_args()

def predict_price(theta0, theta1, km):
	return theta0 + theta1 * km

if __name__ == "__main__":
	try:
		thetas = pd.read_csv("thetas.csv")
		theta0 = thetas.at[0, 'theta0']
		theta1 = thetas.at[0, 'theta1']
	except:
		print("Failed to load thetas.csv.")
		exit()
	if options.graph:
		try:
			data = pd.read_csv("data.csv")
			X = data['km'].values
			Y = data['price'].values
			line = theta0 + theta1 * X
		except:
			print("Failed to load thetas.csv.")
			exit(1)
	prediction = predict_price(theta0, theta1, int(options.kms))
	print("The price for a car with {} km is estimated at {}".format(options.kms, round(prediction)))
	if options.graph:
		axes = plt.axes()
		axes = plt.grid()
		plt.scatter(data['km'].values, data['price'].values)
		plt.plot(X, line, c='r')
		plt.title('Linear Regression for price/mileage')
		plt.xlabel('Mileage')
		plt.ylabel('Price')
		plt.show()