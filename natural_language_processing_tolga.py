#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:32:11 2019

@author: tolga
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # This class is for filtering out necessary words
from nltk.stem.porter import PorterStemmer # This class is for stemming

# Iterate over all rows
corpus = []
for i in range(0,1000):
    # Eliminate punctuation but hold the letters and space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Make all the letters lower
    review = review.lower()
    # Split the words
    review = review.split()
    # Get rid of irrelevant filler words (to reduce the dimensions of sparse matrix)
    # And stem the review getting rid of grammar
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
   
# Creating the Bag of Words model by Sklearn
from sklearn.feature_extraction.text import CountVectorizer    
cv = CountVectorizer(max_features = 1500) # Tokenizer can be done here by inputs
# Create the sparse matrix
X = cv.fit_transform(corpus).toarray()
# Independent variable
y = dataset.iloc[:,1].values
 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   

