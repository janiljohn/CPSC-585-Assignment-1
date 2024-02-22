import csv
import random
import math

class gradientDescent:
	
	# TODO: NORMALIZE DATA
	# TODO: ADD POLYNOMIAL ORDERING
	# TODO: SUMMARIZE RESULTS
	
	def __init__(self, infile: str="", order = 1):
		
		# Define Matrix to use CSV data
		self.mat = []
		# Define feature set, including X0
		self.features = ['X0']
		# Define where the independent features (columns) start
		self.xStart = None
		# Define where the independent features (columns) end
		self.xEnd = None
		# Define the dependent column index
		self.y = None
		# Define WT, don't need to define a W matrix since we only use WT
		# in the gradient descent equation  
		self.wt = []
		# Define a list to hold a random sample set of indicies to use
		self.cases = None
		# If a file is specified, then read CSV entries.
		if infile != "":
			self.getDataFromCsv(infile)
			
	def getDataFromCsv(self, infile: str):
		# Open the CSV to read
		with open(infile, mode='r', newline='') as f:
			# Define a reader 
			reader = csv.reader(f, delimiter=',')
			# Obtain the first row of values, the feature names,
			# and copy them to the features array
			self.features += next(reader)
			# Read the rest of the rows
			for r in reader:
				# Add a row to the data matrix
				self.mat.append(r)
				# Convert the values to floats in the just row added 
				self.mat[-1] = [1] + [float(i) for i in self.mat[-1]]
	
	def setXandY(self, xStart: int = 0, xEnd: int, y: int):
		# Set the starting colomn of the independent features
		self.xStart = xStart
		# Set the ending column of the independent features
		self.xEnd = xEnd
		# Set the index column of the dependent features
		self.y = y
		# Set coeffecients based on the set features being used (0 for now)
		for i in range(xStart, xEnd):
			self.wt.append(0)
	
	def getRandomSet(self, shuffles: int = 1)->list:
		# Generate a range of values ending at the length of our dataset
		self.cases = list(range(0, len(self.mat)))
		# Shuffle the values to ensure a random ordering
		for i in range(shuffles):
			random.shuffle(self.cases)
	
	def predict(self, c: list)->float:
		# Define a predictor value to hold our wT dot X summation
		predicted = 0
		# Go through each coeffiecients
		for j in range(len(self.wt)):
			# Perform each piece of the wT dot X equation
			predicted += (self.wt[j] * c[j])	
		# Return our predicted value
		return predicted
			
	def train(self, alpha: float, batchSize: int):
		# Temp array for the coefficients, allows us to update wT in parallel
		temp = [0 for i in range(len(self.wt))]
		# Go through each coefficient and Xj pair
		for j in range(len(self.wt)):
			# Obtain the (wT dot Xj (Predicted) - Y (Actual)) * Xij piece of GD update rule
			for i in range(batchSize):
				# Pop an index to test from the data set
				c = self.mat[self.cases.pop()]
				# Generate a set of cases incase the case set is empty
				if(len(self.cases) == 0):
					self.getRandomSet(shuffles = 5)
				# (wT dot Xj (Predicted) - Y (Actual)) * Xij
				temp[j] += ((self.predict(c) - c[self.y]) * c[j])
			# Multiply our temp by alpha and 1\N. N in this case is our batch size
			# temp[j] is now alpha * 1/n * sum((predicited - actual) * Xij)
			temp[j] *= (alpha * (1 / batchSize))
		# Update our coeffiecents as per the final part of the update rule
		# Wj = Wj - alpha * 1\N * sum((predicted - actual) * Xij)
		for j in range(len(self.wt)):
			self.wt[j] -= temp[j]
			
	def test(self, testSize: int):
		# Define a total error accumulator
		totErr = 0
		for j in range(testSize):
			# Generate a test case
			c = self.mat[self.cases.pop()]
			# Generate a new set of cases when they are empty
			if(len(self.cases) == 0):
				self.getRandomSet()
			# Add the generated error to the total error
			totErr += (abs(self.predict(c) - c[self.y]) / c[self.y])
		# Output the average error
		return (totErr / testSize)
	
	# Helper function to step through a piece of GD.
	def trainTest(self, alpha: float = 1, batchSize: int = 25, testSize: int = 5)->float:
		self.train(alpha, batchSize)
		return self.test(testSize)
		
	# Gradient descent Handler:
		# Setting Batch Size to 1 results in Stochastic Descent
		# Setting Batch Size to the size of our dataset result in Batch Gradient Descent
		# Anything in between is Minibatch GD.
	def gd(self, alpha: float = 1, batchSize: int = 25, testSize = 5, minErr: float = 0.1):
		# Generate a new set of values
		self.getRandomSet(shuffles = 5)
		avgErr = 1
		# Loop until the average error reachces a certain threshold
		while (avgErr >= minErr):
			# Step through a train/test, recording the average error at each step
			avgErr = self.trainTest(alpha, batchSize, testSize)
		
	#For debugging purposes.
	def showData(self, cutoff: int = 20):
		print(f"\nX Features: {self.features[self.xStart:self.xEnd]}\nY Features: {[self.features[self.y]]}\n")
		for i in range(cutoff):
			print(f"Independent: {self.mat[i][self.xStart:self.xEnd]}\n  Dependent: {[self.mat[i][self.y]]}\n")
		print(f"Coefficients: {self.wt}\n")
			
# TEST RUN
gas_properties = gradientDescent("GasProperties.csv")
gas_properties.setXandY(0, 5, 5)
gas_properties.gd(alpha = 0.0000000000025, batchSize = 10, testSize = 2, minErr =  0.005)
gas_properties.showData()
gas_properties.getRandomSet()
print(gas_properties.test(len(gas_properties.mat)))
