import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

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
        plt.title('Receiver Operating Characteristic (Log Scale)')
        plt.legend(loc="lower right")
        plt.savefig(f"ROC Curve-{classifier_name}.png")
        plt.show()
        
    
    return conf_matrix, accuracy, precision