import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from time import time

from part1 import poly_main, preprocess_data
from utils import format_time


#   Part 1 A 
#       Summary of basic modeling results

# Load the GasProperties.csv file into a DataFrame
gas_properties = pd.read_csv('GasProperties.csv')

preped_data = preprocess_data.preprocess(gas_properties)

# print(gas_properties)
# print(preped_data)

polynomial_orders_tested = [1,2,3,4]

testing_sizes = [
    # 0.05, 0.1, 0.15, 
    0.2 
    # 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
    ]

doingNormalized = False
for iterDataset in [gas_properties, preped_data]:
    for iterTestSize in testing_sizes:
        orders = []
        training_rmses = []
        training_r2s = []
        training_times = [] 
        testing_rmses = [] 
        testing_r2s = []

        for iterOrder in polynomial_orders_tested:
            thisResult = poly_main.polynomial_regression(df= iterDataset, degree=iterOrder, test_size=iterTestSize)
            print(f"===Order: {iterOrder}; Train/Test Split: {iterTestSize}")
            print(thisResult)
            print(f"=========")

            orders.append(f"Order {thisResult['Polynomial order']}")
            training_rmses.append(thisResult["Training RMSE"])
            training_r2s.append(thisResult['Training R^2'])
            training_times.append(thisResult['Training time'])
            testing_rmses.append(thisResult['Testing RMSE'])
            testing_r2s.append(thisResult['Testing R^2'])


        # Create a DataFrame
        results_df = pd.DataFrame({
            'Polynomial order': orders,
            'Training RMSE': training_rmses,
            'Training R^2': training_r2s,
            'Training time': training_times,
            'Testing RMSE': testing_rmses,
            'Testing R^2': testing_r2s
        })

        # Display the table
        print(results_df.to_string(index=False))

        if doingNormalized == False:
            results_df.to_csv(f'GasProperties_results_{int(iterTestSize*100)}_split.csv', index=False)
        else:
            results_df.to_csv(f'GasProperties_preprossesed_results_{int(iterTestSize*100)}_split.csv', index=False)
        doingNormalized = True

# #   Part 1 C 
# #       Get polynomial function

