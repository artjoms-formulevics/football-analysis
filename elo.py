#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:21:14 2021

@author: afo
"""

# old rating of each team 
r_old_1 = 1000
r_old_2 = 1000

# goals by each teams
g_1 = 0
g_2 = 0

# default k
k = 40.0

def calculate_elo(team_1, team_2, r_old_1, r_old_2, g_1, g_2, k=40.0):
    
    # if r_old_1 == 0:
    #     r_old_1 = default_elo
    # if r_old_2 == 0:
    #     r_old_2 = default_elo
    
    
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
    
    
