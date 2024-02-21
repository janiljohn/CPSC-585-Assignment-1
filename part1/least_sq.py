import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from time import time

# Load the GasProperties.csv file into a DataFrame
gas_properties = pd.read_csv('GasProperties.csv')

# Define features and target variable
X = gas_properties[['T', 'P', 'TC', 'SV']]
y = gas_properties['Idx']

# Split the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression object
regressor = LinearRegression()

# Time the training process
start_time = time()
regressor.fit(X_train, y_train)
end_time = time()

# Predict on training and test data
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# Calculate RMSE and R-squared for training and test sets
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

training_time = end_time - start_time

# Print the summary
print(f"Order 1 Polynomial Regression")
print(f"Training RMSE: {train_rmse:.5f}")
print(f"Training R^2: {train_r2:.5f}")
print(f"Training time: {training_time:.4f} seconds")
print(f"Testing RMSE: {test_rmse:.5f}")
print(f"Testing R^2: {test_r2:.5f}")
