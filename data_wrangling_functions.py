#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:27:50 2021

@author: afo
"""

import json
import pandas as pd
import re

from data_gathering_functions import loop_single_match_data

# Data to load fixtures data from json and pre-prcess it
def process_fixtures(p):
    
    # Loading fixtures json
    with open(p+'/data/fixtures.json') as f:
        fixtures = json.load(f)
        
    # Make transformations of dataset / tidying data
    for i in range(0, len(fixtures)):
        
        fixtures[i]["team_h"] = fixtures[i]["h"]["title"]
        fixtures[i]["team_a"] = fixtures[i]["a"]["title"]
        fixtures[i]["id_h"] = fixtures[i]["h"]["id"]
        fixtures[i]["id_a"] = fixtures[i]["a"]["id"]
        fixtures[i]["goals_h"] = int(fixtures[i]["goals"]["h"])
        fixtures[i]["goals_a"] = int(fixtures[i]["goals"]["a"])
        fixtures[i]["xG_h"] = float(fixtures[i]["xG"]["h"])
        fixtures[i]["xG_a"] = float(fixtures[i]["xG"]["a"])
        fixtures[i]["h_forecast_w"] = float(fixtures[i]["forecast"]["w"])
        fixtures[i]["h_forecast_d"] = float(fixtures[i]["forecast"]["d"])
        fixtures[i]["h_forecast_l"] = float(fixtures[i]["forecast"]["l"])
        
        # Remove unwanted fields after transformations
        entries_to_remove = ('goals', 'forecast', 'xG', 'h', 'a')
        for k in entries_to_remove:
            fixtures[i].pop(k, None)
        
    # Loop to go through every fixture and load detailed player stats for each match
    for i in range(0, len(fixtures)):
        
        # Limit to 10 retries
        retries = 0
        got_data = False
        match_id = int(fixtures[i]['id'])
        
        temp, got_data = loop_single_match_data(match_id, retries, got_data, i)  # call the function to get data about match
        
        # If data was received successfully, making some data conversions and sum single player data into a match-level-data
        if got_data == True: 
            temp_h = temp['h']
            temp_a = temp['a']
            
            temp_h = pd.DataFrame.from_dict(temp['h'], orient='index')
            temp_a = pd.DataFrame.from_dict(temp['a'], orient='index')
        
        
            temp_h = temp_h[['shots', 'yellow_card', 'red_card', 'key_passes', 'xA', 'xGChain', 'xGBuildup']]
            temp_h[['shots', 'yellow_card', 'red_card', 'key_passes']] = temp_h[['shots', 'yellow_card', 'red_card', 'key_passes']].astype('int')
            temp_h[['xA', 'xGChain', 'xGBuildup']] = temp_h[['xA', 'xGChain', 'xGBuildup']].astype('float')
               
            temp_a = temp_a[['shots', 'yellow_card', 'red_card', 'key_passes', 'xA', 'xGChain', 'xGBuildup']]
            temp_a[['shots', 'yellow_card', 'red_card', 'key_passes']] = temp_a[['shots', 'yellow_card', 'red_card', 'key_passes']].astype('int')
            temp_a[['xA', 'xGChain', 'xGBuildup']] = temp_a[['xA', 'xGChain', 'xGBuildup']].astype('float')
            
            temp_sum_h = temp_h.sum()
            temp_sum_a = temp_a.sum()
            
            # Split data into columns with prefixes for home and away
            for j in range(0, len(temp_h.sum())):
                
                fixtures[i][temp_h.sum().index[j]+'_h'] = temp_sum_h[j]
                fixtures[i][temp_a.sum().index[j]+'_a'] = temp_sum_a[j]
                    
    return fixtures


# Data to load teams data from json, merge it with fixtures data and pre-process it
def process_teams(p, fixtures):
    # Loading team data json
    with open(p+'/data/teams.json') as f:
        teams = json.load(f)
    
    # Loop through every team
    for team in range(0, len(teams)):
        
        # loop through every fixture
        for i in range(0, len(fixtures)):
            
            # loop through every row
            for j in range(len(teams[team]['history'])):
                
                # if found a team as home team for fixtures data, add details from team table to fixtures table
                if fixtures[i]['id_h'] == teams[team]['id'] and fixtures[i]['datetime'] == teams[team]['history'][j]['date']:
                    
                    fixtures[i]['npxG_h'] = teams[team]['history'][j]['npxG'] 
                    fixtures[i]['ppda_att_h'] = teams[team]['history'][j]['ppda']['att'] 
                    fixtures[i]['ppda_def_h'] = teams[team]['history'][j]['ppda']['def'] 
                    fixtures[i]['deep_h'] = teams[team]['history'][j]['deep'] 
                    fixtures[i]['xpts_h'] = teams[team]['history'][j]['xpts'] 
                    fixtures[i]['pts_h'] = teams[team]['history'][j]['pts'] 
                
                # if not, then it is away team 
                elif fixtures[i]['id_a'] == teams[team]['id'] and fixtures[i]['datetime'] == teams[team]['history'][j]['date']:
                
                    fixtures[i]['npxG_a'] = teams[team]['history'][j]['npxG'] 
                    fixtures[i]['ppda_att_a'] = teams[team]['history'][j]['ppda']['att'] 
                    fixtures[i]['ppda_def_a'] = teams[team]['history'][j]['ppda']['def'] 
                    fixtures[i]['deep_a'] = teams[team]['history'][j]['deep'] 
                    fixtures[i]['xpts_a'] = teams[team]['history'][j]['xpts'] 
                    fixtures[i]['pts_a'] = teams[team]['history'][j]['pts'] 
                    
    return fixtures


# def column_name_tidying(team_list, lst):
    
#     all_teams = list(team_list.keys())  # get the list of all teams in this competition
#     team_set_list = []  # placehodler list
    
#     # loop the data for each team
#     for team in range(0, len(all_teams)):
        
#         # placeholder frames
#         t = pd.DataFrame()
#         t2 = pd.DataFrame()
        
#         team_set = []  # placehodler
        
#         # loop for each season
#         for i in range(0, len(lst)):
            
#             df = lst[i]  # season data
            
#             # slice data for individual team (where it present as home) and rename columns to use suffixes correctly 
#             temp = df.loc[df['team_h'] == all_teams[team]]
#             t = temp.filter(regex=r'(_h$|_a$|datetime)', axis=1)
#             t['location'] = 'h'
#             t = t.reset_index()
#             t = t.rename(columns=lambda x: re.sub('_h$','',x))
#             t = t.rename(columns=lambda x: re.sub('_a$','_vs',x))
            
#             # same suffixing of the columns for away games
#             t2 = df.loc[df['team_a'] == all_teams[team]]
#             t2 = t2.filter(regex=r'(_a$|_h$|datetime)', axis=1)
#             t2['location'] = 'a'
#             t2 = t2.reset_index()
#             t2 = t2.rename(columns=lambda x: re.sub('_h$','',x))
#             t2 = t2.rename(columns=lambda x: re.sub('_a$','_vs',x))
            
#             team_set.append(pd.concat([t,t2]))  # return back slices together
            
#         team_set_list.append(pd.concat(team_set).sort_values(by='datetime'))  # append the data for the full league back together and sort
    
#     return team_set_list
           
# m = a['location'] == 'a'

# a.loc[m, ['team', 'team_vs']] = (
#     df.loc[m, ['team_vs', 'team']].values)