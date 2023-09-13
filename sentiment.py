# importing required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Reading our csv dataset of twitter sentiments.
data = pd.read_csv("D:\Projects\sentiment-analysis\Tweets.csv")

# Getting data and labels from the csv file
text = data["text"]
sentiment = data["sentiment"]

# Required data preprocessing steps
data.dropna()
data['text'].fillna("No text", inplace=True)

# Splitting the data into train and test splits
text_train, text_test, sentiment_train, sentiment_test = train_test_split(text, sentiment, test_size=0.2, random_state=42)

# Using Tfidf Vectorizer for our text in the file
tfidf = TfidfVectorizer(max_features=5000)
text_trained = tfidf.fit_transform(text_train)
text_tested = tfidf.transform(text_test)

# We are using logistic regression model for this case
model = LogisticRegression(max_iter = 2000)
model.fit(text_trained, sentiment_train)

# model Performance section
prediction = model.predict(text_tested)

accuracy = accuracy_score(sentiment_test, prediction)
classification_rep = classification_report(sentiment_test, prediction)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)