#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:08:00 2021

@author: afo
"""

import pandas as pd
import numpy as np
import pickle
import re

# Function to calculate rolling mean of the features
def calculate_lag(df):
    
    df.drop(['yellow_card', 'red_card', 'ppda_att', 
           'ppda_def', 'pts', 'xpts', 'npxG'], axis=1, inplace=True, errors='ignore')
    
    original_number_of_cols = len(df.columns)-1  # get number of columns for processing
    
    # # calculate average of last 5 observations for certain variables
    for i in range(4, original_number_of_cols):
        df[df.iloc[:, i].name + '_last5'] = df.iloc[:, i].rolling(window = 5).mean().shift(1)

    # # calculate average of last 10 observations for certain variables
    # for i in range(6, original_number_of_cols+2):
    #     df[df.iloc[:, i].name + '_last10'] = df.iloc[:, i].rolling(window = 10).mean()
        
    # drop old, unprocessed variables
    df.drop(df.iloc[:, 5:original_number_of_cols-1], inplace=True, axis=1, errors='ignore')
    
    # encode location variable to numerical representation
    df['location'] = np.where(df['location'] == 'h', 1, df['location'])
    df['location'] = np.where(df['location'] == 'a', 0, df['location'])
    
    return df


cleaned_sets = []  # placeholder list
datalist = pickle.load(open("raw_data.pickle", "rb"))  # load data from first step
lists = []

# main loop through every league
for i in range(0, len(datalist)):

    dataset = datalist[i]
    
    temp = pd.concat(dataset)  # get all matches from one league
    
    all_teams = temp['team_h'].unique().tolist()  # get list of team names in each league
    
    # loop through each team
    for team in all_teams:
        
        # filter matches of one team
        t = pd.concat([temp[temp['team_h'] == team], temp[temp['team_a'] == team]])
        t = t.sort_values(by=['datetime'])
    
        # split them to home & away games
        t1 = t[t['team_h'] == team]
        t2 = t[t['team_a'] == team]
        
        # filter unwanted columns
        t1 = t1.filter(regex=r'(_h$|datetime|id$|team_h$|team_a$)', axis=1)
        t2 = t2.filter(regex=r'(_a$|datetime|id$|team_h$|team_a$)', axis=1)
        
        t1.drop(['id_h'], axis=1, inplace=True)
        t2.drop(['id_a'], axis=1, inplace=True)
        
        # encode the location as a new column
        t1['location'] = 'h'
        t2['location'] = 'a'
        
        # replace suffixes 'h' and 'a' in columns to make them unifiable
        for j in range(4, len(t1.columns)):
            t1.rename(columns={ t1.columns[j]: re.sub('_h$','',t1.columns[j]) }, inplace = True)
            t2.rename(columns={ t2.columns[j]: re.sub('_a$','',t2.columns[j]) }, inplace = True)
    
        lists.append(pd.concat([t1,t2]).sort_values(by=['datetime']))  # append new dataframes to the list

# for each item in lists
for i in range(0, len(lists)):    
    df = lists[i].reset_index(drop=True)
    cleaned_sets.append(calculate_lag(df))  # create lagged (rolling mean) data from every season and append to new list

# get single dataset
data = pd.concat(cleaned_sets) # merge data
data = data[data['datetime'] >= '2015-07-01 00:00:00']  # remove first season so that elo rating could converge

# create a unique match id for each match
data['match_id'] = data['team_h']+ data['team_a']  
data['match_id'] = [''.join(sorted(s)) for s in data['match_id']]

temp = data.drop_duplicates(['match_id','datetime', 'location'])  # drop duplicated games

# group all games by match id and datetime. Now each game is unique, but we need to correctly assign who is home team and who is away
ans = [pd.DataFrame(y) for x, y in temp.groupby(['match_id','datetime'], as_index=False)]

for i in range(0, len(ans)):

    ans[i].reset_index(drop=True, inplace=True)
    
    # if the match after removing duplicates is 'home'
    if ans[i].loc[0,['location']].values == 1:
        
        # get home & away teams
        h = ans[i].iloc[0,:]
        a = ans[i].iloc[1,:]
        
        # add suffixes home to home team
        temp_columns1 = h[4:6].add_suffix('_h')
        temp_columns2 = h[7:len(h)-1].add_suffix('_h')
        h = pd.concat([h[0:4], temp_columns1, h[6:7], temp_columns2])
        h.name = 'home'
        
        # add suffixes away to away team
        temp_columns1 = a[4:6].add_suffix('_a')
        temp_columns2 = a[7:len(a)-1].add_suffix('_a')
        a = pd.concat([a[0:4], temp_columns1, a[6:7], temp_columns2])
        a.name = 'away'
    
    # if the match is away, make the other way round
    else: 
            
        h = ans[i].iloc[1,:]
        a = ans[i].iloc[0,:]
    
        temp_columns1 = h[4:6].add_suffix('_h')
        temp_columns2 = h[7:len(h)-1].add_suffix('_h')
        h = pd.concat([h[0:4], temp_columns1, h[6:7], temp_columns2])
        h.name = 'home'
        
        temp_columns1 = a[4:6].add_suffix('_a')
        temp_columns2 = a[7:len(a)-1].add_suffix('_a')
        a = pd.concat([a[0:4], temp_columns1, a[6:7], temp_columns2])
        a.name = 'away'
    
    # combine processed data to unified data item
    ans[i] = pd.concat([h,a], axis = 1)
    ans[i]['home'].fillna(ans[i]['away'], inplace=True)
    del ans[i]['away']
    ans[i] = ans[i].T
    ans[i].reset_index(drop=True, inplace=True)

# combine all the data together and filter unwanted features
data = pd.concat(ans)
data = data.dropna()  # drop NAs
data.drop(['location', 'id', 'datetime', 'team_h', 'team_a'], axis=1, inplace=True, errors='ignore')
data['goals'] = data['goals_h'] - data['goals_a'] 
data.drop(['goals_h', 'goals_a'], axis=1, inplace=True, errors='ignore')
data = data.astype(np.float64)
data.reset_index(drop=True, inplace=True)

#  write to file (just to store)
data.to_csv('data.csv')
