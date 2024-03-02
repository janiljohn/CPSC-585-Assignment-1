import pandas as pd
from bow import bag_of_wordsify, split_dataset
from classifier import train_logistic_regression, train_naive_bayes, evaluate_classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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