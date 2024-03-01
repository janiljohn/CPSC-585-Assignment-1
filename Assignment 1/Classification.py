# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
# import matplotlib.pyplot as plt

# # Load the dataset
# data = pd.read_csv('emails.csv')

# # Split the data into features (X) and target variable (y)
# X = data['text']
# y = data['spam']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize CountVectorizer
# vectorizer = CountVectorizer(stop_words='english', lowercase=True)

# # Fit and transform the training data
# X_train_bow = vectorizer.fit_transform(X_train)

# # Transform the testing data
# X_test_bow = vectorizer.transform(X_test)

# # Initialize Logistic Regression and Naïve Bayes models
# lr = LogisticRegression(max_iter=1000)
# nb = MultinomialNB()

# # Train the models
# lr.fit(X_train_bow, y_train)
# nb.fit(X_train_bow, y_train)

# # Make predictions
# y_pred_lr = lr.predict(X_test_bow)
# y_pred_nb = nb.predict(X_test_bow)

# # Evaluate Logistic Regression model
# print("Logistic Regression:")
# print("Accuracy:", accuracy_score(y_test, y_pred_lr))
# print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# # Evaluate Naïve Bayes model
# print("\nNaïve Bayes:")
# print("Accuracy:", accuracy_score(y_test, y_pred_nb))
# print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# # Compute ROC curve and ROC area for each class
# fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test_bow)[:, 1])
# roc_auc_lr = auc(fpr_lr, tpr_lr)

# fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(X_test_bow)[:, 1])
# roc_auc_nb = auc(fpr_nb, tpr_nb)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
# plt.plot(fpr_nb, tpr_nb, color='darkblue', lw=2, label='Naïve Bayes (AUC = %0.2f)' % roc_auc_nb)
# plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('emails.csv')

# Split the data into features (X) and target variable (y)
X = data['text']
y = data['spam']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words='english', lowercase=True)

# Fit and transform the training data
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_bow = vectorizer.transform(X_test)

# Define a list of models
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Naïve Bayes', MultinomialNB())
]

# Iterate over the models
for name, model in models:
    # Train the model
    model.fit(X_train_bow, y_train)

    # Make predictions
    y_pred = model.predict(X_test_bow)

    # Evaluate the model
    print(f"{name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_bow)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.show()
