import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
	description="Predict a car's price.")
parser.add_argument('kms', help="car's mileage")
parser.add_argument('--graph', '-g', action='store_true', help='show graph')
parser.add_argument('--costs', '-c', action='store_true', help='show costs')
options = parser.parse_args()

def predict_price(theta0, theta1, km):
	return theta0 + theta1 * km

if __name__ == "__main__":
	try:
		thetas = pd.read_csv("thetas.csv")
		theta0 = thetas.at[0, 'theta0']
		theta1 = thetas.at[0, 'theta1']
	except:
		print("Failed to load thetas.csv. Execute training program (linear_regression.py)")
		print("\033[0;31mIn case you wonder, 0 + 0 * km = 0\033[0m")
		exit()
	if options.graph:
		try:
			data = pd.read_csv("data.csv")
			X = data['km'].values
			Y = data['price'].values
			line = theta0 + theta1 * X
		except:
			print("Failed to load data.csv.")
			exit(1)
	if options.costs:
		try:
			data_cost = pd.read_csv("costs.csv")
			iterations = data_cost['iterations'].values
			costs = data_cost['costs'].values
		except:
			print("Failed to load costs.csv.")
			exit(1)
	prediction = predict_price(theta0, theta1, int(options.kms))
	print("The price for a car with {} km is estimated at \033[32;3m{}\033[0m".format(options.kms, round(prediction)))
	if options.graph:
		axes = plt.axes()
		axes = plt.grid()
		plt.scatter(data['km'].values, data['price'].values)
		plt.plot(X, line, c='r')
		plt.title('Linear Regression for price/mileage')
		plt.xlabel('Mileage')
		plt.ylabel('Price')
		plt.show()
	if options.costs:
		axes = plt.axes()
		axes = plt.grid()
		plt.scatter(iterations, costs)
		plt.title('Cost vs Iterations')
		plt.xlabel('Number of Iterations')
		plt.ylabel('Costs')
		plt.show()
		# plt.plot(range(iterations),costs)
		# plt.xlabel('Number of Iterations')
		# plt.ylabel('Costs')
		# plt.title('Cost vs Iterations Analysis')