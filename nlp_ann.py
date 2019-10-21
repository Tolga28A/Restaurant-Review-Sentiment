#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:46:03 2019

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


### Artificia Neural Networks ###
# Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def ann_model():
    # Initializing the ANN
    classifier = Sequential()
    # Add the input and the first hidden layer
    #classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(activation="relu", input_dim=1500, units=6, kernel_initializer="uniform"))
    # Add the second hidden layer
    classifier.add(Dense(activation="relu", input_dim=1500, units=6))
    # Add the output layer
    classifier.add(Dense(activation="sigmoid", input_dim=1500, units=1))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier


# Wrap Keras model so it can be used by scikit-learn
neural_network = KerasClassifier(build_fn=ann_model,epochs=30, batch_size=100,verbose=0)
from sklearn.model_selection import cross_val_score
accur_kf = cross_val_score(neural_network,X_train,y_train,cv=5)
accur_kf_m = accur_kf.mean()
accur_kf_s = accur_kf.std()

"""
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Calculate the accuracy
from sklearn.metrics import accuracy_score
accur = 100*accuracy_score(y_test,y_pred)

# Calculate also the k-fold accuracy
from sklearn.model_selection import cross_val_score
accur_kf = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,scoring="accuracy")
accur_kf_m = accur_kf.mean()
accur_kf_s = accur_kf.std()
"""