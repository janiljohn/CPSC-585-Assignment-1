# //////// LEAST SQUARE METHOD ////////
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time

# Load the GasProperties dataset from GasProperties.csv
GasProperties = pd.read_csv('GasProperties.csv')

# X should contain the features T, P, TC, SV, and y should contain Idx
X = GasProperties[['T', 'P', 'TC', 'SV']]
y = GasProperties['Idx']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the maximum polynomial order
max_order = 5

# Initialize lists to store results
orders = []
train_rmse = []
train_r2 = []
train_times = []
test_rmse = []
test_r2 = []

# Iterate over polynomial orders
for order in range(1, max_order + 1):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=order)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Fit a linear regression model
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X_train_poly, y_train)
    end_time = time.time()

    # Predict on training and testing data
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Calculate RMSE and R^2 for training and testing data
    train_rmse_value = mean_squared_error(y_train, y_train_pred, squared=False)
    train_r2_value = r2_score(y_train, y_train_pred)
    test_rmse_value = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2_value = r2_score(y_test, y_test_pred)

    # Store results
    orders.append(order)
    train_rmse.append(train_rmse_value)
    train_r2.append(train_r2_value)
    train_times.append(end_time - start_time)
    test_rmse.append(test_rmse_value)
    test_r2.append(test_r2_value)

# Create a summary table
summary_table = pd.DataFrame({
    'Polynomial Order': orders,
    'Training RMSE': train_rmse,
    'Training R^2': train_r2,
    'Training Time (s)': train_times,
    'Testing RMSE': test_rmse,
    'Testing R^2': test_r2
})

print(summary_table)









# //////// GRADIENT DESCENT METHOD ////////
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# Load the GasProperties dataset from GasProperties.csv
GasProperties = pd.read_csv('GasProperties.csv')

# Normalize the features
X = GasProperties[['T', 'P', 'TC', 'SV']]
X = (X - X.mean()) / X.std()
X['bias'] = 1

# Convert the DataFrame to a numpy array for easier computation
X = X.to_numpy()
y = GasProperties['Idx'].to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the maximum polynomial order
max_order = 5

# Initialize lists to store results
orders = []
train_rmse = []
train_r2 = []
train_times = []
test_rmse = []
test_r2 = []

# Set hyperparameters for gradient descent
alpha = 0.01  # learning rate
epochs = 1000  # number of iterations

# Iterate over polynomial orders
for order in range(1, max_order + 1):
    start_time = time.time()
    
    # Generate polynomial features for training data
    X_poly_train = np.c_[X_train[:, :-1], np.ones_like(X_train[:, -1])]  # exclude bias term
    for i in range(2, order + 1):
        X_poly_train = np.c_[X_poly_train, np.power(X_train[:, :-1], i)]

    # Initialize the weights randomly
    np.random.seed(42)
    weights = np.random.rand(X_poly_train.shape[1])

    # Gradient Descent
    for _ in range(epochs):
        # Calculate predictions
        y_pred = np.dot(X_poly_train, weights)

        # Calculate the error
        error = y_pred - y_train

        # Calculate the gradient
        gradient = np.dot(X_poly_train.T, error) / len(y_train)

        # Update weights
        weights -= alpha * gradient

    end_time = time.time()

    # Calculate RMSE and R^2 for training data
    y_train_pred = np.dot(X_poly_train, weights)
    train_rmse_value = mean_squared_error(y_train, y_train_pred, squared=False)
    train_r2_value = r2_score(y_train, y_train_pred)

    # Store results for training data
    orders.append(order)
    train_rmse.append(train_rmse_value)
    train_r2.append(train_r2_value)
    train_times.append(end_time - start_time)

    # Generate polynomial features for testing data
    X_poly_test = np.c_[X_test[:, :-1], np.ones_like(X_test[:, -1])]
    for i in range(2, order + 1):
        X_poly_test = np.c_[X_poly_test, np.power(X_test[:, :-1], i)]

    # Calculate predictions for testing data
    y_test_pred = np.dot(X_poly_test, weights)

    # Calculate RMSE and R^2 for testing data
    test_rmse_value = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2_value = r2_score(y_test, y_test_pred)

    # Store results for testing data
    test_rmse.append(test_rmse_value)
    test_r2.append(test_r2_value)

# Create a summary table
summary_table = pd.DataFrame({
    'Polynomial Order': orders,
    'Training RMSE': train_rmse,
    'Training R^2': train_r2,
    'Training Time (s)': train_times,
    'Testing RMSE': test_rmse,
    'Testing R^2': test_r2
})

print(summary_table)









# //////// LASSO METHOD ////////
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time

# Load the GasProperties dataset from GasProperties.csv
GasProperties = pd.read_csv('GasProperties.csv')

# Split the data into features (X) and target variable (y)
X = GasProperties[['T', 'P', 'TC', 'SV']]
y = GasProperties['Idx']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the maximum polynomial order
max_order = 5

# Initialize lists to store results
orders = []
train_rmse = []
train_r2 = []
train_times = []
test_rmse = []
test_r2 = []

# Iterate over polynomial orders
for order in range(1, max_order + 1):
    start_time = time.time()

    # Create a pipeline with polynomial features and Lasso regression
    model = make_pipeline(
        PolynomialFeatures(degree=order),
        Lasso(alpha=0.1)  # alpha is the regularization strength
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate RMSE and R^2 for training and testing data
    train_rmse_value = mean_squared_error(y_train, y_train_pred, squared=False)
    train_r2_value = r2_score(y_train, y_train_pred)
    test_rmse_value = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2_value = r2_score(y_test, y_test_pred)

    # Store results
    orders.append(order)
    train_rmse.append(train_rmse_value)
    train_r2.append(train_r2_value)
    test_rmse.append(test_rmse_value)
    test_r2.append(test_r2_value)

    end_time = time.time()
    train_times.append(end_time - start_time)

# Create a summary table
summary_table = pd.DataFrame({
    'Polynomial Order': orders,
    'Training RMSE': train_rmse,
    'Training R^2': train_r2,
    'Training Time (s)': train_times,
    'Testing RMSE': test_rmse,
    'Testing R^2': test_r2
})

print(summary_table)
