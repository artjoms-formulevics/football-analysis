#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 00:10:29 2021

@author: afo
"""

import asyncio
import json

import aiohttp

from understat import Understat

import nest_asyncio
nest_asyncio.apply()

async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_league_players(
            "epl", 2018,
            player_name="Paul Pogba",
            team_title="Manchester United"
        )
        print(json.dumps(player))

loop = asyncio.get_event_loop()
loop.run_until_complete(main())