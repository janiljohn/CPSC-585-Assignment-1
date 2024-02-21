import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from time import time

def polynomial_regression(df, degree=1, test_size=0.2):
    """
    Perform polynomial regression on a DataFrame.

    Parameters:
    df (DataFrame): Pandas DataFrame containing the dataset with a target column named 'Idx'.
    degree (int): The degree of the polynomial regression.

    Returns:
    dict: A dictionary containing the regression summary (RMSE, R^2, training time) for both training and testing sets.
    """
    
    # Define features and target variable
    X = df.drop('Idx', axis=1)
    y = df['Idx']

    # Generate polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Split the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=0)

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

    # Compile the results
    results = {
        'Polynomial order': degree,
        'Training RMSE': train_rmse,
        'Training R^2': train_r2,
        'Training time': training_time,
        'Testing RMSE': test_rmse,
        'Testing R^2': test_r2
    }

    return results
