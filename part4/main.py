import pandas as pd
# creating the feature matrix 
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv('data.csv', encoding='ISO-8859-1');

matrix = CountVectorizer(max_features=1000)
X = matrix.fit_transform(dataset).toarray()