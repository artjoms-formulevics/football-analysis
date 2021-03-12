#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:38:37 2021

@author: afo
"""

import asyncio
import json
from os.path import abspath
from inspect import getsourcefile
import os
import aiohttp
import time
import sys

from understat import Understat

import nest_asyncio
nest_asyncio.apply()


# Get the working dir
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

# Main asynch loop for data gathering from Understat
loop = asyncio.get_event_loop()


# Function to collect fixtures from a given year in given league and save to json
async def collect_fixtures(year, league):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        fixtures = await understat.get_league_results(
            league, year)
        with open(p+'/data/fixtures.json', 'w') as fp:
            json.dump(fixtures, fp)
        
# Function to collect players from a given year in given league and save to json            
async def collect_league_players(year, league):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        players = await understat.get_league_players(
            league,
            year
           # team_title="Liverpool"
        )
        with open(p+'/data/players.json', 'w') as fp:
            json.dump(players, fp)
            
# Function to get details about the player stats of a match given its id  
async def get_player_data(match_id):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        players = await understat.get_match_players(match_id)
        
        return players

# Function to collect team data from a given year in given league and save to json
async def get_teams(year, league):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
                    
        teams = await understat.get_teams(
            league,
            year
        )
        with open(p+'/data/teams.json', 'w') as fp:
            json.dump(teams, fp)

# Function to try to get the fixtures from Understat, limited to retry number
def loop_fixtures(league, year, retries):
    while retries <= 10:
        try:
            loop.run_until_complete(collect_fixtures(year, league))
            print("Fixtures for " + league + ", " + str(year) + " collected succesfully!")
            time.sleep(1)
            break
        except UnboundLocalError:
            retries = retries + 1
            if retries == 10:
                print("Failed on Fixtures loop!")
                sys.exit()

# Function to try to get the players from Understat, limited to retry number
def loop_players(league, year, retries):
            while retries <= 10:
                try:
                    loop.run_until_complete(collect_league_players(year, league))
                    print("League players for " + league + ", " + str(year) + " collected succesfully!")
                    time.sleep(1)
                    break
                except UnboundLocalError:
                    retries = retries + 1
                    if retries == 10:
                        print("Failed on League players loop!")
                        sys.exit()
                        
# Function to try to get the teams from Understat, limited to retry number                     
def loop_teams(league, year, retries):
            while retries <= 10:
                try:
                    loop.run_until_complete(get_teams(year, league))
                    print("Team list for " + league + ", " + str(year) + " collected succesfully!")
                    time.sleep(1)
                    break
                except UnboundLocalError:
                    retries = retries + 1
                    if retries == 10:
                        print("Failed on Team list loop!")
                        sys.exit()
                        
                        
def loop_single_match_data(match_id, retries, got_data, i):
            while retries < 10:
                try:
                    temp = loop.run_until_complete(get_player_data(match_id))
                    got_data = True
                    break
                except UnboundLocalError:
                    retries = retries + 1
                    if retries == 10:
                        got_data = False
                        temp = ''
                        print("Failed on Player Match data loop! Failed on number nr: " + str(i))
                        pass
                
            return temp, got_data