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

from understat import Understat

import nest_asyncio
nest_asyncio.apply()

# Get the working dir
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

loop = asyncio.get_event_loop()

async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        fixtures = await understat.get_league_results(
            "epl", 2019)
        with open(p+'/data/fixtures.json', 'w') as fp:
            json.dump(fixtures, fp)
        
        players = await understat.get_league_players(
            "epl",
            2019
           # team_title="Liverpool"
        )
        with open(p+'/data/players.json', 'w') as fp:
            json.dump(players, fp)
            
        # teams = await understat.get_teams(
        #     "epl",
        #     2019,
        #     title="Liverpool"
        # )
        # with open(p+'/data/teams.json', 'w') as fp:
        #     json.dump(teams, fp)
        # teams = await understat.get_teams(
        #     "epl",
        #     2019)
        # with open(p+'/data/teams.json', 'w') as fp:
        #     json.dump(teams, fp)
        #print(json.dumps(fixtures))
        
async def get_player_data(match_id):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        players = await understat.get_match_players(match_id)
        # with open(p+'/data/temp.json', 'w') as fp:
        #     json.dump(players, fp)
        
        return players

async def get_teams():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
                    
        teams = await understat.get_teams(
            "epl",
            2019
            #title="Liverpool"
        )
        with open(p+'/data/teams.json', 'w') as fp:
            json.dump(teams, fp)
            


loop.run_until_complete(main())
loop.run_until_complete(get_teams())

with open(p+'/data/fixtures.json') as f:
    fixtures = json.load(f)
    
# with open(p+'/data/players.json') as f:
#     players = json.load(f)

# with open(p+'/data/teams.json') as f:
#     teams = json.load(f)

# entries_to_remove = ('id', 'short_title')
# for k in entries_to_remove:

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
    
    entries_to_remove = ('goals', 'forecast', 'xG', 'h', 'a')
    for k in entries_to_remove:
        fixtures[i].pop(k, None)
    

# df = pd.DataFrame(fixtures)

# df['player_stats_h'] = ""
# df['player_stats_a'] = ""
# player_stats_h_place = df.columns.get_loc('player_stats_h')
# player_stats_a_place = df.columns.get_loc('player_stats_a')

# df['shots'] = ""
# df['yellow_card'] = ""
# df['red_card'] = ""
# df['key_passes'] = ""
# df['xA'] = ""
# df['xGChain'] = ""
# df['xGBuildup'] = ""



with open(p+'/data/players.json') as f:
    players = json.load(f)

# temp_h[0]

# new_keys = list(range(0,len(temp_h.keys())))

# for key,n_key in zip(temp_h.keys(), new_keys):
#     temp_h[n_key] = temp_h.pop(key)

# dict_with_ints = dict((k,to_int(v)) for k,v in temp_h[0].items())
    
# keyfunc = lambda x: x['Name']
# groups = it.groupby(sorted(temp_h, key=keyfunc), keyfunc)
# [{'id':k, 'shots':sum(x['shots'] for x in g)} for k, g in groups]


#player_data = loop.run_until_complete(get_player_data(11643))

for i in range(0, len(fixtures)):
    
    temp = loop.run_until_complete(get_player_data(int(fixtures[i]['id'])))
    temp_h = temp['h']
    temp_a = temp['a']
    
    temp_h = pd.DataFrame.from_dict(temp['h'], orient='index')
    temp_a = pd.DataFrame.from_dict(temp['a'], orient='index')

    # df.iat[i,player_stats_h_place] = temp_h
    # df.iat[i,player_stats_a_place] = temp_a

    temp_h = temp_h[['shots', 'yellow_card', 'red_card', 'key_passes', 'xA', 'xGChain', 'xGBuildup']]
    temp_h[['shots', 'yellow_card', 'red_card', 'key_passes']] = temp_h[['shots', 'yellow_card', 'red_card', 'key_passes']].astype('int')
    temp_h[['xA', 'xGChain', 'xGBuildup']] = temp_h[['xA', 'xGChain', 'xGBuildup']].astype('float')
       
    temp_a = temp_a[['shots', 'yellow_card', 'red_card', 'key_passes', 'xA', 'xGChain', 'xGBuildup']]
    temp_a[['shots', 'yellow_card', 'red_card', 'key_passes']] = temp_a[['shots', 'yellow_card', 'red_card', 'key_passes']].astype('int')
    temp_a[['xA', 'xGChain', 'xGBuildup']] = temp_a[['xA', 'xGChain', 'xGBuildup']].astype('float')
    
    temp_sum_h = temp_h.sum()
    temp_sum_a = temp_a.sum()
    
    for j in range(0, len(temp_h.sum())):
        
        #df.iloc[i, df.columns.get_loc('shots')+j] = temp_h.sum()[j]
        fixtures[i][temp_h.sum().index[j]+'_h'] = temp_sum_h[j]
        fixtures[i][temp_a.sum().index[j]+'_a'] = temp_sum_a[j]




with open(p+'/data/teams.json') as f:
    teams = json.load(f)

for team in range(0, len(teams)):
    
    for i in range(0, len(fixtures)):
        
        for j in range(len(teams[team]['history'])):
        
            if fixtures[i]['id_h'] == teams[team]['id'] and fixtures[i]['datetime'] == teams[team]['history'][j]['date']:
                
                fixtures[i]['npxG_h'] = teams[team]['history'][j]['npxG'] 
                fixtures[i]['ppda_att_h'] = teams[team]['history'][j]['ppda']['att'] 
                fixtures[i]['ppda_def_h'] = teams[team]['history'][j]['ppda']['def'] 
                fixtures[i]['deep_h'] = teams[team]['history'][j]['deep'] 
                fixtures[i]['xpts_h'] = teams[team]['history'][j]['xpts'] 
                fixtures[i]['pts_h'] = teams[team]['history'][j]['pts'] 
            
                #datetime.strptime(fixtures[i]['datetime'], '%Y-%m-%d %H:%M:%S') == datetime.strptime(teams[team]['history'][j]['date'], '%Y-%m-%d %H:%M:%S')
                
            elif fixtures[i]['id_a'] == teams[team]['id'] and fixtures[i]['datetime'] == teams[team]['history'][j]['date']:
            
                fixtures[i]['npxG_a'] = teams[team]['history'][j]['npxG'] 
                fixtures[i]['ppda_att_a'] = teams[team]['history'][j]['ppda']['att'] 
                fixtures[i]['ppda_def_a'] = teams[team]['history'][j]['ppda']['def'] 
                fixtures[i]['deep_a'] = teams[team]['history'][j]['deep'] 
                fixtures[i]['xpts_a'] = teams[team]['history'][j]['xpts'] 
                fixtures[i]['pts_a'] = teams[team]['history'][j]['pts'] 
            

df = pd.DataFrame(fixtures)
#df = df[df.isnull().sum(axis=1) < 7]
