#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

directory = 'nfl_data'

games = pd.read_csv(f'{directory}/games.csv')
plays = pd.read_csv(f'{directory}/plays.csv')
players = pd.read_csv(f'{directory}/players.csv')
player_play = pd.read_csv(f'{directory}/player_play.csv')
#tracking1 = pd.read_csv(f'{directory}/tracking_week_1.csv')
tracking3 = pd.read_csv(f'{directory}/tracking_week_3.csv')

# filter player_play for plays when motion at ball snap
#motion_plays = player_play[player_play['inMotionAtBallSnap'] == True]
motion_plays = player_play[player_play['motionSinceLineset'] == True]

# join with players to get position info
motion_plays = motion_plays.merge(
    players[['nflId', 'position', 'displayName']], 
    on='nflId', 
    how='inner'
)

print(len(plays))

# filter out penalty plays
plays_wo_penalty = plays[plays['playNullifiedByPenalty'] == 'N']
print(len(plays_wo_penalty))

# filter to only plays with man cover 1
man_coverage = plays_wo_penalty[(plays_wo_penalty['pff_manZone'] == 'Man') &
                                (plays_wo_penalty['pff_passCoverage'] == 'Cover-1')]
print(len(man_coverage))


# join with plays to get play-specific info
motion_plays = motion_plays.merge(
    man_coverage[['playId', 'gameId', 'possessionTeam', 'defensiveTeam', 'yardlineNumber',
           'absoluteYardlineNumber', 'offenseFormation', 'receiverAlignment', 
           'yardsGained', 'expectedPointsAdded', 'pff_runConceptPrimary',
           'pff_runPassOption', 'pff_passCoverage', 'pff_manZone'
           ]], 
    on='playId', 
    how='inner'
)

# join with tracking data to get player tracking info
motion_plays = motion_plays.merge(tracking3, on=['playId', 'nflId'], how='inner')

# save to new csv
motion_plays.to_csv("preprocessed_motion.csv", index=False)