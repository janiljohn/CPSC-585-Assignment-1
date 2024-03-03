import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#TEMP IMPORTS
from sklearn.svm import SVC








# BOW FILE 
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









# CLASSIFIER FILE
def train_logistic_regression(X_train, y_train, poly_degree=1):
    # Transform input data to include polynomial features
    poly = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly.fit_transform(X_train)
    
    # Initialize the Logistic Regression classifier
    logistic_regression_classifier = LogisticRegression()
    # Train the classifier with polynomial features
    logistic_regression_classifier.fit(X_train_poly, y_train)
    return logistic_regression_classifier, poly

def train_naive_bayes(X_train, y_train, poly_degree=1):
    # Transform input data to include polynomial features
    poly = PolynomialFeatures(degree=poly_degree)
    X_train_poly = poly.fit_transform(X_train)
    
    # Initialize the Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB()
    # Train the classifier with polynomial features
    naive_bayes_classifier.fit(X_train_poly, y_train)
    return naive_bayes_classifier, poly

def evaluate_classifier(classifier, poly, X_test, y_test, classifier_name):
    # Transform the test data to include polynomial features
    X_test_poly = poly.transform(X_test)
    
    # Predict the labels for the test set with polynomial features
    y_pred = classifier.predict(X_test_poly)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate the precision
    precision = precision_score(y_test, y_pred, zero_division=0)
    
    # Calculate the ROC curve and AUC for classifiers that support probability predictions
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test_poly)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve with log-scaled y-axis
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.01, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {classifier_name}')
        plt.legend(loc="lower right")
        plt.savefig(f"ROC Curve-{classifier_name}.png")
        plt.show()
        
    
    return conf_matrix, accuracy, precision







# MAIN FILE
# Example custom feature functions
def contains_not(text):
    return 1 if 'not' in text.split() else 0

def contains_awesome(text):
    return 1 if 'awesome' in text.split() else 0

#Load the email dataset
dataset = pd.read_csv('emails.csv', encoding='ISO-8859-1')
feature_functions = [contains_not, contains_awesome]
X_featureless, feature_names = bag_of_wordsify(dataset=dataset,feature_functions=[], max_token_features=50)
X, feature_names = bag_of_wordsify(dataset=dataset,feature_functions=feature_functions, max_token_features=50)

y = dataset['spam']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Example Degree
poly_degree = 1

logistic_regression_classifier, lr_poly = train_logistic_regression(X_train, y_train, poly_degree=poly_degree)
naive_bayes_classifier, nb_poly = train_naive_bayes(X_train, y_train, poly_degree=poly_degree)

# Evaluating the classifiers
print("Logistic Regression Classifier Evaluation")
lr_conf_matrix, lr_accuracy, lr_precision = evaluate_classifier(logistic_regression_classifier, lr_poly, X_test, y_test, classifier_name="Logistic_regression")

print("Confusion Matrix:\n", lr_conf_matrix)
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)

print("\nNaive Bayes Classifier Evaluation")
nb_conf_matrix, nb_accuracy, nb_precision = evaluate_classifier(naive_bayes_classifier, nb_poly, X_test, y_test, classifier_name="Naive_Bayes")

print("Confusion Matrix:\n", nb_conf_matrix)
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)