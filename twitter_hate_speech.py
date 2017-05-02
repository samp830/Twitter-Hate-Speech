import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn import linear_model, datasets, model_selection
from sklearn.cross_validation import cross_val_score
import re

#Read CSV
text = pd.read_csv('twitter.csv', encoding = "ISO-8859-1")
#Default UTF-8 encoding was returning an error, ISO is fine to use because dataset is in English
#Selecting Relevant Columns
columns_of_interest = ['does_this_tweet_contain_hate_speech','does_this_tweet_contain_hate_speech:confidence', 'tweet_text']
text = text[columns_of_interest]

#Cleaning
def clean(row):
    cleaned = ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([x][0-9]+)|([0-9]+)'," ",row).split())
    return cleaned
text['tweet_text'] = text['tweet_text'].apply(clean)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Initialize Vectorizer
vect = TfidfVectorizer()
#vect = CountVectorizer() (0.84785005512679157 Logreg score vs 0.85005512679162076)
vect = TfidfVectorizer(ngram_range=(1, 2))
#intuition being that bi-gram can distinguish hate speech from offensive language, however there is tradeoff of adding more noise in hopes that it will help signal
#If using bigram change min df to 2, have at least twice
#Can tune stop-words, ngrams, max_df, min_df
regex_match = '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([x][0-9]+)|([0-9]+)'
#gets rid of Twitter handles, punctuation, urls, 'x89s etc., and all numbers
x = str(list(text["tweet_text"]))
#corpus of all words in training set
corpus = (' '.join(re.sub(regex_match," ",x).split()))
#Filter out stopwords
from nltk.corpus import stopwords
filtered_words = [word for word in corpus.split() if word not in stopwords.words('english')]
#Fit Vectorizer
# vect.fit(corpus.split())
vect.fit(filtered_words)
#encode
text['does_this_tweet_contain_hate_speech'] = text.does_this_tweet_contain_hate_speech.map({'The tweet uses offensive language but not hate speech':-1, 'The tweet is not offensive':-1, 'The tweet contains hate speech':1})
# define X and y
X = text.tweet_text
weights = text['does_this_tweet_contain_hate_speech:confidence']
y = text.does_this_tweet_contain_hate_speech
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, w_train, w_test, y_train, y_test = train_test_split(X, weights, y)
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)
def my_init(shape, dtype=None):
    return w_train;
from keras.models import Sequential
twitter_model = Sequential()
from keras.layers import Dense
from keras.layers import Activation
twitter_model.add(Dense(64, activation='tanh', input_shape=(X_train_dtm.shape[1],)))
twitter_model.add(Dense(32, activation='tanh'))
twitter_model.add(Dense(16, activation='tanh'))
twitter_model.add(Dense(8, activation='tanh'))
twitter_model.add(Dense(1, activation='tanh'))
#twitter_model.add(Activation('tanh'))
twitter_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
print(twitter_model.summary())
# Changing from sparse matrix to dense
A = X_train_dtm
A = A.todense()
twitter_model.fit(A, y_train, epochs=3, sample_weight=w_train)
# Changing from sparse matrix to dense
B = X_test_dtm
B = B.todense()
# Evaluating
print(twitter_model.evaluate(B, y_test, sample_weight=w_test))
# Weighted NN: 83.41% accuracy
