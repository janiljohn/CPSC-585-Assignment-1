import csv
import random
import time

class gradientDescent:
	
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
		self.trnQueue = None
		self.testCases = None
		self.trained = None
		self.tested = None
		# Define the polynomial order
		self.order = 1
		# Define the results
		self.ymean = 0
		self.rss = 0
		self.tss = 0
		self.mse = 0
		self.rmse = 0
		self.rsq = 0
		self.totPred = 0
		self.elapsedTime = 0
		self.trainCounter = 0
		self.testCounter = 0
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
	
	def getRandomSet(self, split: int, shuffles: int = 1)->list:
		# Generate a range of values ending at the length of our dataset
		cases = list(range(0, len(self.mat)))
		# Shuffle the values to ensure a random ordering
		for i in range(shuffles):
			random.shuffle(cases)
		# Split those cases into a train and test side
		self.trnQueue = [] + cases[:split]
		self.tstQueue = [] + cases[split:]
		self.trained = []
		self.tested = []
	
	# Switches the queues and already used values 
	def resetTrnCases(self):
		self.trnQueue = [] + self.trained
		self.trained = []
		# shuffle our queue again
		random.shuffle(self.trnQueue)
		
	def resetTstCases(self):
		self.tstQueue = [] + self.tested
		self.tested = []
		random.shuffle(self.tstQueue)
			
	def normal(self, feature: float, fi: int)->float:
		# Xnew = (Xold - Xmin) / (Xmax - Xmin)
		return (feature - self.normVals[fi][0]) / (self.normVals[fi][1] - self.normVals[fi][0])
		# return feature
	
	def unnormal(self, feature: float, fi: int)->float:
		# Xold = (Xnew) * (Xmax - Xmin) + Xmin
		return ((feature)*(self.normVals[fi][1] - self.normVals[fi][0])) + self.normVals[fi][0]
		# return feature
	
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
				if(len(self.trnQueue) == 0):
					self.resetTrnCases()
				# Pop an index to test from the data set, then normalize it
				ci = self.trnQueue.pop()
				self.trained.append(ci)
				c = self.normalizer([]+self.mat[ci])
				# (wT dot Xj (Predicted) - Y (Actual)) * Xij
				temp[j] += alpha * (1 / batchSize) * ((self.predict(c) - c[self.y]) * (c[j] ** self.order))
			# Multiply our temp by alpha and 1\N. N in this case is our batch size
			# temp[j] is now alpha * 1/n * sum((predicited - actual) * Xij)
		# Update our coeffiecents as per the final part of the update rule
		# Wj = Wj - alpha * 1\N * sum((predicted - actual) * Xij)
		for j in range(len(self.wt)):
			self.wt[j] -= temp[j]
		self.trainCounter += batchSize
			
	def test(self, testSize: int):
		# Define a total error accumulator
		totErr = 0
		for j in range(testSize):
			# Generate a test case, then normalize it.
			if(len(self.tstQueue) == 0):
				self.resetTstCases()
			ci = self.tstQueue.pop()
			self.tested.append(ci)
			c = self.normalizer([]+self.mat[ci])
			# Generate a new set of cases when they are empty
			# Unnormalize the dependent features
			predicted = self.unnormal(self.predict(c), self.y)
			actual = self.unnormal(c[self.y], self.y)
			# Add the generated error to the total error
			totErr += (abs(predicted - actual) / actual)
		# Output the average error
		self.testCounter += testSize
		return (totErr / testSize)
	
	# Helper function to step through a piece of GD.
	def trainTest(self, alpha: float = 1, batchSize: int = 25, testSize: int = 5)->float:
		self.train(alpha, batchSize)
		return self.test(testSize)
	
	def getYMean(self):
		# Record the mean of our predicted values
		self.ymean = self.totPred / len(self.mat)
	
	def getRss(self, caseSet: list):
		self.rss = 0
		self.totPred = 0
		# Go through the cases in a set
		for row in caseSet:
			# Pop off a case, then normalize it
			c = self.normalizer([]+self.mat[row])
			# Get the the true value of our actual
			actual = self.unnormal(c[self.y], self.y)
			# Get the true value of our predicted
			predicted = self.unnormal(self.predict(c), self.y)
			# Record the total predicted values, will use this in another function
			# (there is probably a better way to do this... idc!)
			self.totPred += predicted
			# Get the rss for one case, the add it to the total
			self.rss += ((actual - predicted) ** 2)
	
	# USE AFTER YOU GET THE RSS
	def getTss(self, caseSet: list):
		self.tss = 0
		# Get the mean of our total predicted
		self.getYMean()
		# Go through the cases in a set
		for row in caseSet:
			# Pop a case, the normalize it
			c = self.normalizer([]+self.mat[row])
			# Get the true value of the actual
			actual = self.unnormal(c[self.y], self.y)
			# Record the TSS for an instance
			self.tss += ((actual - self.ymean) ** 2)
			
	def getMse(self, caseSize):
		self.mse = self.rss / caseSize
	
	def getRmse(self):
		self.rmse = self.mse ** 0.5
	
	def getRsq(self):
		self.rsq = 1 - (self.rss / self.tss)
		
	# Gradient descent Handler:
		# Setting Batch Size to 1 results in Stochastic Descent
		# Setting Batch Size to the size of our dataset result in Batch Gradient Descent
		# Anything in between is Minibatch GD.
	def gd(self, alpha: float = 1, order: float = 1, batchSize: int = 50, testSize = 10, minErr: float = 0.1, split: int = 0):
		self.trainCounter = 0
		self.testCounter = 0
		self.wt = []
		for i in range(self.xStart, self.xEnd):
			self.wt.append(random.random())
		# Srart training timer
		start = time.time()
		# Generate a new set of values
		self.order = order
		self.getRandomSet(split, shuffles = 5)
		avgErr = 1
		# Loop until the average error reachces a certain threshold
		while (avgErr >= minErr):
			# Step through a train/test, recording the average error at each step
			avgErr = self.trainTest(alpha, batchSize, testSize)
		# End training timer
		end = time.time()
		# Record the total time
		self.elapsedTime = (end - start)
		
	#For debugging purposes.
	def showData(self, cutoff: int = 20):
		print(f"\nX Features: {self.features[self.xStart:self.xEnd]}\nY Features: {[self.features[self.y]]}\n")
		for i in range(cutoff):
			print(f"Independent: {self.mat[i][self.xStart:self.xEnd]}\n  Dependent: {[self.mat[i][self.y]]}\n")
	
	def getResults(self)->list:
		trainCases = [] + self.trnQueue + self.trained
		testCases = [] + self.tstQueue + self.tested
		self.getRss(trainCases)
		self.getMse(len(trainCases))
		self.getRmse()
		self.getTss(trainCases)
		self.getRsq()
		trainRmse = self.rmse
		trainRsq = self.rsq
		self.getRss(testCases)
		self.getMse(len(testCases))
		self.getRmse()
		self.getTss(testCases)
		self.getRsq()
		testRmse = self.rmse
		testRsq = self.rsq
		self.tstQueue = [] + testCases
		self.tested = []
		return [self.order] + ([] + self.wt) +  [trainRmse, trainRsq, self.trainCounter, self.elapsedTime, testRmse, testRsq, self.testCounter, self.test(len(testCases))]
		
# TEST RUN
gp = gradientDescent("GasProperties.csv")
gp.setXandY(0, 5, 5)
testResults = []
results_format = ["ORDER"] + gp.features[:gp.xEnd] + ["TRAIN RMSE", "TRAIN R^2", "TRAINED CASES", "TRAINING TIME", "TEST RMSE", "TEST R^2", "TESTED CASES", "ERROR"]
print("Generating models...")
with open("GP_MODELS.csv", mode = 'w', newline='') as outfile:
	writer = csv.writer(outfile)
	writer.writerow(results_format)
	for i in range(1, 21):
		gp.gd(alpha = 0.01, order = float(i/2), batchSize = 500, testSize = 100, minErr = 0.0033, split = int(len(gp.mat) * (5/6)))
		writer.writerow(gp.getResults())
print("Done!")
