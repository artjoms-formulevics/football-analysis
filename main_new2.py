#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 00:01:20 2021

@author: afo
"""

import asyncio
import json
import traceback
from os.path import abspath
from inspect import getsourcefile
import os
import aiohttp
import pandas as pd
from datetime import datetime
import itertools as it
import statistics
import re
import numpy as np
import time
import sys
import pickle

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import parallel_backend
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

from keras.models import load_model

from understat import Understat

from elo import calculate_elo, add_elo_ratings
from helper_functions import print_full
from daata_gathering_functions import collect_fixtures, collect_league_players, get_player_data, get_teams
from data_gathering_functions import loop_fixtures, loop_players, loop_teams, loop_single_match_data
from data_wrangling_functions import process_fixtures, process_teams, column_name_tidying

from data_preprocessing import clean_set

import nest_asyncio
nest_asyncio.apply()


# Get the working dir
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

# Main asynch loop for data gathering from Understat
loop = asyncio.get_event_loop()

# listing all years and leagues to query
years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
leagues = ['epl', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1', 'rfpl']


datalist = []  # list to store data from each league

# Main loop to query each league
for league in leagues:
    
    # holder-lists for each type of data
    lst = []
    all_teams = []
    fixtures = []
    players = []
    team_set = []
    team_set_list = []
    teams = []
    
    # Query each year for each league
    for year in years:
        
        df = None  # placehodler df
        
        # try to collect fixtures from league, retrying up to 10 times, in case of network/other failures
        retries = 0
        loop_fixtures(league, year, retries)
            
        # try to collect players from league, retrying up to 10 times, in case of network/other failures    
        retries = 0
        loop_players(league, year, retries)
        
        # try to collect teams from league, retrying up to 10 times, in case of network/other failures    
        retries = 0
        loop_teams(league, year, retries)
        
        # Loading player json (currently not used)
        # with open(p+'/data/players.json') as f:
        #     players = json.load(f)
        
        # Load & Process data on fixtures
        fixtures = process_fixtures(p)
        
        # Load & Process data on teams & merge with fixtures
        fixtures = process_teams(p, fixtures)
                        
        # conver fixtures to pandas df and append to new list holder
        df = pd.DataFrame(fixtures)
        lst.append(df)

        # calculate and add elo rating to data
        lst, team_list = add_elo_ratings(lst)

    # Call the function, tidying the dataset, normalize column names
    team_set_list = column_name_tidying(team_list, lst)
    
    
    # Append data for league into main list of leagues
    datalist.append(team_set_list)
    print("League " + league + " collected succesfully!")


# save processed data to drive
pickle.dump(datalist, open("raw_data.pickle", "wb") )

cleaned_sets = []
datalist = pickle.load(open("raw_data.pickle", "rb"))

for i in range(0, len(datalist)):

    df = datalist[i]
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    
    cleaned_sets.append(clean_set(df))


data = pd.concat(cleaned_sets)
data.to_csv('data.csv')

data = pd.read_csv('data.csv', index_col=0)

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

sc = StandardScaler()
X_train.loc[:, X_train.columns != 'location'] = sc.fit_transform(X_train.loc[:, X_train.columns != 'location'] )  
X_test.loc[:, X_test.columns != 'location'] = sc.transform(X_test.loc[:, X_test.columns != 'location'])  

# mms = MinMaxScaler()
# y_train = mms.fit_transform(pd.DataFrame(y_train))
# y_test = mms.transform(pd.DataFrame(y_test))

# y_train = y_train.astype(int)
# y_test = y_test.astype(int)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# y_train = y_train.tolist()
# y_train = np.array([np.array(xi) for xi in y_train])

# y_test = y_test.tolist()
# y_test = np.array([np.array(xi) for xi in y_test])

def baseline_model():
    # create model
    model = keras.Sequential()
    model.add(layers.Dense(20, input_dim=len(data.columns)-1, activation='relu', kernel_initializer='normal',  kernel_regularizer='l1'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(5, activation='relu'))  #,  kernel_regularizer='l2'
    model.add(layers.Dropout(0.10))
    model.add(layers.Dense(1, activation='linear'))
    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.001)
   #  'categorical_crossentropy' mean_squared_error
   # tf.keras.metrics.MeanAbsoluteError()
   # tf.keras.metrics.Accuracy()
   #tf.keras.metrics.CategoricalAccuracy
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    model.summary()
    
    return model

# KerasClassifier //  KerasRegressor

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=1, verbose=1,
                           validation_data=(X_test, y_test)) #callbacks=[es]

# kfold = KFold(n_splits=5)
# results = cross_val_score(estimator, X_train, y_train, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X_train, y_train)


y_true, y_pred = y_test, estimator.predict(X_test)

# temp = []
# for i in range(0,len(y_true)):
#     temp.append(np.argmax(y_true[i]))
# y_true = pd.DataFrame(temp)
# y_true.columns = ['actual']
# y_pred = pd.DataFrame(y_pred)
# y_pred.columns = ['rounded_pred']
# comparison = y_true.join(y_pred)

# y_true = mms.inverse_transform(y_true)
# y_pred = mms.inverse_transform(y_pred)

outcome = y_true
outcome = [1 if x>0  else x for x in outcome]
outcome = [-1 if x<0  else x for x in outcome]
outcome = [0 if x==0  else x for x in outcome]

pred_outcome = np.round(y_pred, decimals=0)
pred_outcome = [1 if x>0  else x for x in pred_outcome]
pred_outcome = [-1 if x<0  else x for x in pred_outcome]
pred_outcome = [0 if x==0  else x for x in pred_outcome]

d = {'actual':y_true,'rounded_pred': np.round(y_pred, decimals=0), 'pred':y_pred, 'outcome':outcome, 'pred_outcome':pred_outcome}
comparison = pd.DataFrame.from_dict(d).reset_index()

# comparison = pd.DataFrame(y_true, columns=["home_true", "away_true"]).join(pd.DataFrame(y_pred, 
#                         columns=["home_pred", "away_pred"])).join(pd.DataFrame(np.round(y_pred, decimals=0), 
#                             columns=["home_round", "away_round"]))
                                                                               
# outcome = (comparison['home_true'] - comparison['away_true']).tolist()    
# outcome = [1 if x>0  else x for x in outcome]
# outcome = [-1 if x<0  else x for x in outcome]
# outcome = [0 if x==0  else x for x in outcome]

# pred_outcome = (comparison['home_pred'] - comparison['away_pred']).tolist()  
# pred_outcome = [1 if x>0  else x for x in pred_outcome]
# pred_outcome = [-1 if x<0  else x for x in pred_outcome]
# pred_outcome = [0 if x==0  else x for x in pred_outcome]   

# d = {'outcome':outcome,'pred_outcome':pred_outcome}
# comparison = comparison.join(pd.DataFrame(d))                                                      

print(comparison)

print("MSE: %0.5f" % mean_squared_error(y_true, y_pred))
print("MAE: %0.5f" % mean_absolute_error(y_true, y_pred))
print("R2: %0.5f" % r2_score(y_true, y_pred))

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
#print("ROC AUC: %0.3f" % roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo'))


# reg = LinearRegression().fit(X_train, y_train)
# y_true, y_pred = y_test, reg.predict(X_test)


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
#print("ROC AUC: %0.3f" % roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo'))



estimator.model.save('saved_model.h5')
model2 = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=10, verbose=1)
model2.model = load_model('saved_model.h5')
