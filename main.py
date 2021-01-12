#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:54:56 2021

@author: afo
"""

import pandas as pd
import os
import traceback
from os.path import abspath
from inspect import getsourcefile
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt

# Get the working dir
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

database = p + '/database.sqlite'
conn = sqlite3.connect(database)

# tables = pd.read_sql("""SELECT *
#                         FROM sqlite_master
#                         WHERE type='table';""", conn)
                        
countries = pd.read_sql("""SELECT *
                        FROM Country;""", conn)
                        
leagues = pd.read_sql("""SELECT *
                        FROM League;""", conn)
                        
teams = pd.read_sql("""SELECT *
                        FROM Team;""", conn)

matches = pd.read_sql("""SELECT *
                        FROM Match;""", conn)

player = pd.read_sql("""SELECT *
                        FROM Player;""", conn)

teams_attributes = pd.read_sql("""SELECT *
                        FROM Team_Attributes;""", conn)

player_attributes = pd.read_sql("""SELECT *
                        FROM Player_Attributes;""", conn)


df = pd.read_sql("""SELECT *
                        FROM Player_Attributes, 
                        JOIN Player ON Player.player_api_id = Player_Attributes.player_api_id;""", conn)

conn.close()


del player["id"]
del player_attributes["id"]
del matches["id"]

df = pd.merge(player_attributes, player, how='left', left_on='player_api_id', right_on='player_api_id')
temp = df.sample(5)

match_players = matches.iloc[:, 54:76]
match_players[match_players.isnull().any(1)]
match_players[match_players.isnull().all(1)]



