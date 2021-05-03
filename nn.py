#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:18:32 2021

@author: afo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

# function to perform PCA dimensionality reduction on training and test sets 
def PCA_reduction(X_train, X_test, cols, n=0.90):
    
    # fit PCA on training set and plot curve
    pca = PCA(n_components=n).fit(X_train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    
    # transfrom train and test sets
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    # get number of components
    n_pcs= pca.components_.shape[0]
    
    # get the index of the most important feature on each component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

    # get the names
    most_important_names = [cols[most_important[i]] for i in range(n_pcs)]
    
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    
    # build the dataframe of most important components
    most_important_features = pd.DataFrame(dic.items())
    
    print(most_important_features)  # see the results
    
    return X_train, X_test

# function to split the data into two sets and scale it
def data_split_and_scale(data):
    
    train, test = train_test_split(data, test_size=0.1)
    
    ## Split to features and classes
    X_train = train.drop(['goals'], axis=1, errors='ignore')
    y_train = train['goals']
    
    y_train = y_train.to_list()
    y_train = np.array([np.array(xi) for xi in y_train])
    
    X_test = test.drop(['goals'], axis=1, errors='ignore')
    y_test = test['goals']
    
    y_test = y_test.to_list()
    y_test = np.array([np.array(xi) for xi in y_test])
    
    cols = list(X_test.columns)  # store original column list
    
    # Scale data
    sc = StandardScaler()
    X_train.loc[:, X_train.columns != 'location'] = sc.fit_transform(X_train.loc[:, X_train.columns != 'location'] )  
    X_test.loc[:, X_test.columns != 'location'] = sc.transform(X_test.loc[:, X_test.columns != 'location'])  
    
    X_train, X_test = PCA_reduction(X_train, X_test, cols, 0.95)  # call dim reduction function

    return X_train, y_train, X_test, y_test

# defining model 
def keras_model():
    
    model = keras.Sequential()
    model.add(layers.Dense(100, input_dim=col_number, activation='relu', kernel_initializer='normal', kernel_regularizer='l2'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(40, activation='relu'))  #,  kernel_regularizer='l2'
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.keras.metrics.MeanAbsoluteError()])  # Compile model
    model.summary()
    
    return model

# Function to analyze and print the results
def analyze_model(y_test, y_true):
    
    # get the outcomes to compare (if it was a win, lose or draw for home team based on goal difference)
    outcome = y_true
    outcome = [1 if x>0  else x for x in outcome]
    outcome = [-1 if x<0  else x for x in outcome]
    outcome = [0 if x==0  else x for x in outcome]
    
    pred_outcome = np.round(y_pred, decimals=0)
    pred_outcome = [1 if x>0  else x for x in pred_outcome]
    pred_outcome = [-1 if x<0  else x for x in pred_outcome]
    pred_outcome = [0 if x==0  else x for x in pred_outcome]
    
    # create an overview df, containing actual data & predictions
    d = {'actual':y_true,'rounded_pred': np.round(y_pred, decimals=0), 'pred':y_pred, 'outcome':outcome, 'pred_outcome':pred_outcome}
    comparison = pd.DataFrame.from_dict(d).reset_index()
                                               
    print(comparison)
    
    # print regression metrics
    print("MSE: %0.5f" % mean_squared_error(y_true, y_pred))
    print("MAE: %0.5f" % mean_absolute_error(y_true, y_pred))
    print("R2: %0.5f" % r2_score(y_true, y_pred))

    # print information for a classification problem, given rounded predictions
    print()
    print("Classfification Report:")
    print()
    print(classification_report(comparison['actual'].astype(int), comparison['rounded_pred'].astype(int)))
    print()
    print("Confusion Matrix:")
    print()
    print(confusion_matrix(comparison['actual'].astype(int), comparison['rounded_pred'].astype(int)))
    print()
    print("Accuracy: %0.3f" % accuracy_score(comparison['actual'].astype(int), comparison['rounded_pred'].astype(int)))
    print("Balanced Accuracy: %0.3f" % balanced_accuracy_score(comparison['actual'].astype(int), comparison['rounded_pred'].astype(int)))
    print("Precision: %0.3f" % precision_score(comparison['actual'].astype(int), comparison['rounded_pred'].astype(int), average='weighted'))
    print("Recall: %0.3f" % recall_score(comparison['actual'].astype(int), comparison['rounded_pred'].astype(int), average='weighted'))
    print("F1 Score: %0.3f" % f1_score(comparison['actual'].astype(int), comparison['rounded_pred'].astype(int), average='weighted'))

    # print information for a classification problem, given rounded predictions (3-way outcomes) 
    print()
    print("Classfification Report:")
    print()
    print(classification_report(comparison['outcome'].astype(int), comparison['pred_outcome'].astype(int)))
    print()
    print("Confusion Matrix:")
    print()
    print(confusion_matrix(comparison['outcome'].astype(int), comparison['pred_outcome'].astype(int)))
    print()
    print("Accuracy: %0.3f" % accuracy_score(comparison['outcome'].astype(int), comparison['pred_outcome'].astype(int)))
    print("Balanced Accuracy: %0.3f" % balanced_accuracy_score(comparison['outcome'].astype(int), comparison['pred_outcome'].astype(int)))
    print("Precision: %0.3f" % precision_score(comparison['outcome'].astype(int), comparison['pred_outcome'].astype(int), average='weighted'))
    print("Recall: %0.3f" % recall_score(comparison['outcome'].astype(int), comparison['pred_outcome'].astype(int), average='weighted'))
    print("F1 Score: %0.3f" % f1_score(comparison['outcome'].astype(int), comparison['pred_outcome'].astype(int), average='weighted'))

    return comparison

# Load previously saved data
data = pd.read_csv('data.csv', index_col=0)

# get the train & test scaled datasets
X_train, y_train, X_test, y_test = data_split_and_scale(data)

# Get number of features
col_number = X_train.shape[1]

# Define model
estimator = KerasRegressor(build_fn=keras_model, epochs=300, batch_size=10, verbose=1,
                           validation_data=(X_test, y_test)) #callbacks=[es]

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

estimator.fit(X_train, y_train)  # fit model

y_true, y_pred = y_test, estimator.predict(X_test)  # make predictions

comparison = analyze_model(y_pred, y_true)  # print the model results and get the overview

# save the model to file
estimator.model.save('saved_model.h5')

# load model from file
# model2 = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=10, verbose=1)
# model2.model = load_model('saved_model.h5')s
