import pandas as pd

def preprocess(inputData: pd.DataFrame):
    data = inputData.copy()  # Work on a copy of the input DataFrame to avoid modifying the original data
    unique_values_count = data.nunique()
    
    # Set the threshold for unique values to consider a field as categorical
    unique_values_threshold = 20
    
    # Extract the names of fields presumed to be categorical based on the threshold
    presumed_categorical_features = unique_values_count[unique_values_count <= unique_values_threshold].index.tolist()
    
    # Print the names of presumed categorical fields
    print("Presumed categorical fields based on the threshold of 20 unique values:")
    print(presumed_categorical_features)
    
    for feature in presumed_categorical_features:
        # Get the unique categories for the feature
        unique_categories = data[feature].unique()
        
        # Create a dictionary that maps each unique category to a unique integer
        category_to_integer_map = {category: index for index, category in enumerate(unique_categories)}
        
        # Replace each category in the DataFrame with its corresponding integer value
        data[feature] = data[feature].map(category_to_integer_map)
        
        # Change the data type to unsigned integer for efficiency
        data[feature] = data[feature].astype('uint8')

    return data