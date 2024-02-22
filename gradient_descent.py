import csv
import random
import math

class gradientDescent:
	
	# TODO: SUMMARIZE RESULTS
	
	def __init__(self, infile: str=""):
		
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
		# Define an array to hold normalizing values
		self.normVals = [[1,1]]
		# Define a list to hold a random sample set of indicies to use
		self.cases = None
		# Define the polynomial order
		self.order = 1
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
				# Account for X0 in our dataset
				self.mat.append([1])
				# Loop through each item in the row
				for items in range(len(list(r))):
					# Convert the items to floats
					temp = float(r[items])
					# Add them to the dataset
					self.mat[-1].append(temp)
					# Through each row, check if the item is a min or a max value, then set them
					# accordingly.
					try:
						self.normVals[items+1][0] = min(temp, self.normVals[items+1][0])
						self.normVals[items+1][1] = max(temp, self.normVals[items+1][1])
					# This block accounts for the start of row processing, as nothing has been
					# added to our normalizing values. 
					except:
						self.normVals.append([temp, temp])
					
	
	def setXandY(self, xStart: int = 0, xEnd: int = 0, y: int = 0):
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
			
	def normal(self, feature: float, fi: int)->float:
		# Xnew = (Xold - Xmin) / (Xmax - Xmin)
		return (feature - self.normVals[fi][0]) / (self.normVals[fi][1] - self.normVals[fi][0])
	
	def unnormal(self, feature: float, fi: int)->float:
		# Xold = (Xnew) * (Xmax - Xmin) + Xmin
		return ((feature)*(self.normVals[fi][1] - self.normVals[fi][0])) + self.normVals[fi][0]
	
	# Call normal function on entire feature set
	def normalizer(self, c: list):
		for i in range(1, len(c)):
			c[i] = self.normal(c[i], i)
		return c
	
	# Call unnormal function on entire feature set
	def unnormalizer(self, c: list):
		for i in range(1, len(c)):
			c[i] = self.unnormal(c[i], i)
		return c
	
	def predict(self, c: list)->float:
		# Define a predictor value to hold our wT dot X summation
		predicted = 0
		# Go through each coeffiecients
		for j in range(len(self.wt)):
			# Perform each piece of the wT dot X equation
			predicted += (self.wt[j] * (c[j] ** self.order))	
		# Return our predicted value
		return predicted
			
	def train(self, alpha: float, batchSize: int):
		# Temp array for the coefficients, allows us to update wT in parallel
		temp = [0 for i in range(len(self.wt))]
		# Go through each coefficient and Xj pair
		for j in range(len(self.wt)):
			# Obtain the (wT dot Xj (Predicted) - Y (Actual)) * Xij piece of GD update rule
			for i in range(batchSize):
				# Pop an index to test from the data set, then normalize it
				c = self.normalizer([]+self.mat[self.cases.pop()])
				# Generate a set of cases incase the case set is empty
				if(len(self.cases) == 0):
					self.getRandomSet(shuffles = 5)
				# (wT dot Xj (Predicted) - Y (Actual)) * Xij
				temp[j] += ((self.predict(c) - c[self.y]) * (c[j] ** self.order))
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
			# Generate a test case, then normalize it.
			c = self.normalizer([]+self.mat[self.cases.pop()])
			# Generate a new set of cases when they are empty
			if(len(self.cases) == 0):
				self.getRandomSet()
			# Unnormalize the dependent features
			predicted = self.unnormal(self.predict(c), self.y)
			actual = self.unnormal(c[self.y], self.y)
			# Add the generated error to the total error
			totErr += (abs(predicted - actual) / actual)
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
	def gd(self, alpha: float = 1, order: float = 1, batchSize: int = 25, testSize = 5, minErr: float = 0.1):
		# Generate a new set of values
		self.order = order
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
		print(f"Order: {self.order}\n")
			
# TEST RUN
gp = gradientDescent("GasProperties.csv")
gp.setXandY(0, 5, 5)
gp.gd(alpha = 0.01, order = 0.5, batchSize = 10, testSize = 10, minErr =  0.005)
gp.showData()
gp.getRandomSet()
print(f"ACCURACY: {(1 - gp.test(len(gp.mat))) * 100}%\n")
custom_case = [1, 255, 20, 0.03, 390, 54]
print(f"CASE: {custom_case[0:5]}\nIDX: {gp.unnormal(gp.predict(gp.normalizer(custom_case)), gp.y)}")
