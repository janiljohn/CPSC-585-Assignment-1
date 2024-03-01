import pandas as pd
from bow import bag_of_wordsify, split_dataset
from classifier import train_logistic_regression, train_naive_bayes, evaluate_classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#Load the email dataset
dataset = pd.read_csv('emails.csv', encoding='ISO-8859-1')

# Assuming that the label column in your dataset is named 'label'
X, feature_names = bag_of_wordsify(dataset)
y = dataset['spam']

# Split the data
X_train, X_test, y_train, y_test = split_dataset(X, y)

# You can now print the feature names or the shapes of the split data
print(feature_names)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Training the classifiers
logistic_regression_classifier = train_logistic_regression(X_train, y_train)
naive_bayes_classifier = train_naive_bayes(X_train, y_train)

# Evaluating the classifiers
print("Logistic Regression Classifier Evaluation")
lr_conf_matrix, lr_accuracy, lr_precision = evaluate_classifier(logistic_regression_classifier, X_test, y_test)

print("Confusion Matrix:\n", lr_conf_matrix)
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)

print("\nNaive Bayes Classifier Evaluation")
nb_conf_matrix, nb_accuracy, nb_precision = evaluate_classifier(naive_bayes_classifier, X_test, y_test)

print("Confusion Matrix:\n", nb_conf_matrix)
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
