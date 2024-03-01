import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve, auc
import matplotlib.pyplot as plt

def train_logistic_regression(X_train, y_train):
    # Initialize the Logistic Regression classifier
    logistic_regression_classifier = LogisticRegression()
    # Train the classifier
    logistic_regression_classifier.fit(X_train, y_train)
    return logistic_regression_classifier

def train_naive_bayes(X_train, y_train):
    # Initialize the Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB()
    # Train the classifier
    naive_bayes_classifier.fit(X_train, y_train)
    return naive_bayes_classifier

def evaluate_classifier(classifier, X_test, y_test):
    # Predict the labels for the test set
    y_pred = classifier.predict(X_test)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate the precision
    precision = precision_score(y_test, y_pred)
    
    # Calculate the ROC curve and AUC
    y_prob = classifier.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve with log-scaled y-axis
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.01, 1.05])
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Log Scale)')
    plt.legend(loc="lower right")
    plt.show()
    
    return conf_matrix, accuracy, precision

