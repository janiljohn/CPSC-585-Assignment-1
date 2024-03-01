import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def bag_of_wordsify(dataset):
    nltk.download('punkt')
    nltk.download('stopwords')

    # Initialize the list for cleaned texts
    cleaned_texts = []

    # Preprocess the text data
    stop_words = set(stopwords.words('english'))
    for text in dataset['text']:
        # Remove non-alphabetic characters and convert to lower case
        text = re.sub('[^A-Za-z]', ' ', text).lower()
        # Tokenize
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word not in stop_words]
        # Join the words back into one string separated by space
        cleaned_text = ' '.join(words)
        cleaned_texts.append(cleaned_text)

    # Vectorize the cleaned text data
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(cleaned_texts).toarray()

    # Return the vectorized data and the feature names
    return X, vectorizer.get_feature_names_out()

def split_dataset(X, y):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

dataset = pd.read_csv('emails.csv', encoding='ISO-8859-1')
bag_of_wordsify(dataset)