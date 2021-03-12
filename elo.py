#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:21:14 2021

@author: afo
"""

import statistics

# old rating of each team 
r_old_1 = 1000
r_old_2 = 1000

# goals by each teams
g_1 = 0
g_2 = 0

# default k
k = 40.0

""" Function calculates elo rating two teams after a game, based on the inputs:
    current rating of each teams
    goals scored by each team
    parameter k (complexity of competition), by default 40 and only used as this number
"""
def calculate_elo(team_1, team_2, r_old_1, r_old_2, g_1, g_2, k=40.0):
    

    # rating diff
    d_r_1 = r_old_1 - r_old_2
    
    # compare goals to determine results
    if g_1 > g_2:
        w = 1.0  # home win
    elif g_1 < g_2:
        w = 0.0  # away win
    else:
        w = 0.5  # draw
    
    # win expectanies
    w_e_1 = 1/(pow(10,(-d_r_1/400))+1)
    w_e_2 = 1 - w_e_1
    
    def calculate_k(goals_home, goals_away):
        
        k = 40.0
        
        if goals_home == goals_away + 2:
            k = 1.5 * k
        elif goals_home == goals_away + 3:
            k = 1.75 * k
        elif goals_home > goals_away + 3:
            k = (1.75 + (goals_home - goals_away - 3)/8) * k
        else:
            k = k
            
        return k
    
    if w == 1.0:
        k = calculate_k(g_1, g_2)
        r_new_1 = r_old_1 + k * (w - w_e_1)
        r_new_2 = r_old_2 + k * (1-w - w_e_2)
    
    elif w == 0.0:
        k = calculate_k(g_2, g_1)
        r_new_1 = r_old_1 + k * (w - w_e_1)
        r_new_2 = r_old_2 + k * (1-w - w_e_2)
        
    else:
        k = k  
        r_new_1 = r_old_1 + k * (w - w_e_1)
        r_new_2 = r_old_2 + k * (1-w - w_e_2)
    
    
    return r_new_1, r_new_2
    
# Function to calculate elo after each fixture and keep track of changes in teams elo rating
def add_elo_ratings(lst):
        
    # main loop to go through all seasons
        for i in range(0, len(lst)):
        
            df = lst[i]
            
            # placehodler columns
            df['elo_h'] = 0 
            df['elo_a'] = 0 
            
            # if its first available season, assign elo of 1500 to all teams
            if i == 0:
                default_elo = 1500.0
                team_list_initial = df['team_h'].unique().tolist()
                team_list = dict(zip(team_list_initial, [default_elo]* len(team_list_initial)))
            
                # if not, assign default elo as average of worst 3 teams from newcomers from second league
                # for remaining teams, assign last elo rating from previous season
            else:
                default_elo = statistics.mean(sorted(set(team_list.values()))[0:3])
                team_list_initial = df['team_h'].unique().tolist()
                team_list_initial = dict(zip(team_list_initial, [default_elo]* len(team_list_initial)))
                team_list_initial.update(team_list)
                team_list = team_list_initial
            
            
            k = 40.0  # default competition rating k
            
            # go throgh all fixtures in a season & calculate elo changes after each match
            for j in range(0, len(df)):
            
                team_h = df.loc[j,'team_h']
                team_a = df.loc[j,'team_a']
                
                elo_h = team_list[team_h]
                elo_a = team_list[team_a]
                
                goals_h = df.loc[j,'goals_h']
                goals_a = df.loc[j,'goals_a']
                
                df.loc[j,'elo_h'] = elo_h
                df.loc[j,'elo_a'] = elo_a
                
                # calculate new elo rating
                new_elo_h, new_elo_a = calculate_elo(team_h, team_a, elo_h, elo_a, goals_h, goals_a, k)
                
                team_list[team_h] = new_elo_h
                team_list[team_a] = new_elo_a
            
            # put the data with elo ratings into original list and return it back
            lst[i] = df
            
        return lst, team_list  # also returns the team list of all the teams ever participating in the competiton in the observable period