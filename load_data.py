#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:32:10 2021

@author: afo
"""


import asyncio
from os.path import abspath
from inspect import getsourcefile
import os
import pandas as pd
import pickle

from elo import add_elo_ratings
from data_gathering_functions import loop_fixtures, loop_players, loop_teams
from data_wrangling_functions import process_fixtures, process_teams #, column_name_tidying

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


    # team_set_list = column_name_tidying(team_list, lst)
    
    
    # Append data for league into main list of leagues
    datalist.append(lst)
    print("League " + league + " collected succesfully!")

# save processed data to drive
pickle.dump(datalist, open("raw_data.pickle", "wb") )