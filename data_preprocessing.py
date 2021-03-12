#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:08:00 2021

@author: afo
"""

import pandas as pd
import numpy as np
import pickle

# Function to pre-process the data before training the model
def clean_set(df):
    
    # Drop unwanted fields
    df.drop(['red_card', 'red_card_vs', 'npxG', 'npxG_vs', 'xGChain', 'xGChain_vs',
             'xGBuildup', 'xGBuildup_vs', 'xA', 'xA_vs', 'ppda_att', 'ppda_def', 'ppda_att_vs', 'ppda_def_vs'], inplace=True, axis=1)
    
    # df.drop(['pts', 'xpts', 'pts_vs', 'xpts_vs'], inplace=True, axis=1)
    
    original_number_of_cols = len(df.columns)-3  # get number of columns for processing
    
    # calculate rolling mean for certain variables
    # for i in range(6, original_number_of_cols):
    #     df[df.iloc[:, i].name + '_mean'] = df.iloc[:, i].expanding().mean()
    
    # calculate average of last 5 observations for certain variables
    for i in range(6, original_number_of_cols+2):
        df[df.iloc[:, i].name + '_last5'] = df.iloc[:, i].rolling(window = 5).mean()
    
    # calculate average of last 10 observations for certain variables
    # for i in range(6, original_number_of_cols+2):
    #     df[df.iloc[:, i].name + '_last10'] = df.iloc[:, i].rolling(window = 10).mean()
        
    
    # get goal difference to be used as a feature
    df['goals'] = df['goals'] - df['goals_vs']
    
    # drop old, unprocessed variables
    df.drop(df.iloc[:, 7:original_number_of_cols], inplace=True, axis=1)
    df.drop(df.iloc[:, 0:6], inplace=True, axis=1)
    
    # encode location variable to numerical representation
    df['location'] = np.where(df['location'] == 'h', 1, df['location'])
    df['location'] = np.where(df['location'] == 'a', 0, df['location'])
    df['location'] = df['location'].astype(int)
    
    df = df.dropna()  # drop NAs
   
    df['goals'] = df['goals'].astype('float')  # convert target to float to use as continous variable
    
    return df



cleaned_sets = []  # placeholder list
datalist = pickle.load(open("raw_data.pickle", "rb"))  # load data from first step

# main loop through every league
for i in range(0, len(datalist)):

    df = datalist[i]
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    
    cleaned_sets.append(clean_set(df))  # clean data from every season and append to new list

# merge data and write to file (just to store)
data = pd.concat(cleaned_sets)
data.to_csv('data.csv')

