import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np


def bag_of_wordsify(dataset, feature_functions=[], max_token_features=1000):
    nltk.download('punkt')
    nltk.download('stopwords')

    # Initialize the list for cleaned texts
    cleaned_texts = []

    # Initialize a list of lists for custom features
    custom_features = [[] for _ in feature_functions]

    # Preprocess the text data
    stop_words = set(stopwords.words('english'))
    for text in dataset['text']:
        # Remove non-alphabetic characters and convert to lower case
        text = re.sub('[^A-Za-z]', ' ', text).lower()
        # Tokenize
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word not in stop_words]
        # Join the words back into one string separated by space, and append to cleaned_texts
        cleaned_text = ' '.join(words)
        cleaned_texts.append(cleaned_text)

        # Apply each custom feature function to the original text (not cleaned)
        for i, func in enumerate(feature_functions):
            custom_features[i].append(func(text))

    # Vectorize the cleaned text data
    vectorizer = CountVectorizer(max_features=max_token_features)
    _thing = vectorizer.fit_transform(cleaned_texts)
    X = _thing.toarray()

    # Convert custom features to a numpy array and add them to the BoW matrix
    for feature in custom_features:
        feature_array = np.array(feature).reshape(-1, 1)
        X = np.hstack((X, feature_array))

    # Update the feature names to include custom features
    feature_names = np.append(vectorizer.get_feature_names_out(), 
                              ['custom_feature_' + str(i) for i in range(len(feature_functions))])

    # Return the updated vectorized data and the updated feature names
    return X, feature_names

def split_dataset(X, y):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

