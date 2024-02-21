import csv
import random

class gradientDescent:
	
	def __init__(self, infile: str=""):
		self.mat = []
		self.features = ['X0']
		self.xStart = None
		self.xEnd = None
		self.y = None
		self.wt = []
		if infile != "":
			self.getDataFromCsv(infile)
			
	def getDataFromCsv(self, infile: str):
		with open(infile, mode='r', newline='') as f:
			reader = csv.reader(f, delimiter=',')
			self.features += next(reader)
			for r in reader:
				self.mat.append(r)
				self.mat[-1] = [1] + [float(i) for i in self.mat[-1]]
	
	def setXandY(self, xStart: int, xEnd: int, y: int):
		self.xStart = xStart
		self.xEnd = xEnd
		self.y = y
		for i in range(xEnd):
			self.wt.append(0)
	
	def getRandomSet(self)->list:
		temp = list(range(0, len(self.mat)))
		random.shuffle(temp)
		return temp
	
	#def gd(self, alpha: float = 1, batchSize: int = 1, minErr):
		
		
	#For debugging purposes.
	def showData(self, cutoff: int = 20):
		print(f"\nX Features: {self.features[self.xStart:self.xEnd]}\nY Features: {[self.features[self.y]]}\n")
		for i in range(cutoff):
			print(f"Independent: {self.mat[i][self.xStart:self.xEnd]}\n  Dependent: {[self.mat[i][self.y]]}\n")
		print(f"Coefficients: {self.wt}\n")
			
gas_properties = gradientDescent("GasProperties.csv")
gas_properties.setXandY(0, 5, 5)
gas_properties.showData()
