from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from time import time

gas_properties = pd.read_csv('../GasProperties.csv')
X = gas_properties.drop('Idx', axis=1)
y = gas_properties['Idx']

alpha_rate = [0.3, 0.2, 0.1]

def runLasso(alpha_rate=0.1): 
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize LASSO model
    lasso_model = Lasso(alpha_rate)  # Adjust alpha as needed

    # Train the model
    start_time = time()
    lasso_model.fit(X_train, y_train)
    end_time = time()

    y_train_pred = lasso_model.predict(X_train)
    y_test_pred = lasso_model.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2 = r2_score(y_test, y_test_pred)

    training_time = end_time - start_time

    results = {
        'Lambda': alpha_rate,
        'Training RMSE': train_rmse,
        'Training R^2': train_r2,
        'Training time': training_time,
        'Testing RMSE': test_rmse,
        'Testing R^2': test_r2
    }

    return results

lambda_rate = []
training_rmses = []
training_r2s = []
training_times = [] 
testing_rmses = [] 
testing_r2s = []

for alpha in alpha_rate:
    thisResult = runLasso(alpha_rate=alpha)
    print(f"===Lambda: {alpha}; Train/Test Split: {0.2}")
    print(thisResult)
    print(f"=========")

    lambda_rate.append(f"Order {thisResult['Lambda']}")
    training_rmses.append(thisResult["Training RMSE"])
    training_r2s.append(thisResult['Training R^2'])
    training_times.append(thisResult['Training time'])
    testing_rmses.append(thisResult['Testing RMSE'])
    testing_r2s.append(thisResult['Testing R^2'])

results_df = pd.DataFrame({
    'Lambda': lambda_rate,
    'Training RMSE': training_rmses,
    'Training R^2': training_r2s,
    'Training time': training_times,
    'Testing RMSE': testing_rmses,
    'Testing R^2': testing_r2s
})

print(results_df.to_string(index=False))

# # Make predictions on the test set
# lasso = lasso_model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, lasso)
# print("Mean Squared Error:", mse)

# # Print the coefficients
# print("Coefficients:", lasso_model.coef_)
