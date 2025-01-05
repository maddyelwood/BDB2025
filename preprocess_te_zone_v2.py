#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

directory = 'nfl_data'

players = pd.read_csv(f'{directory}/players.csv')
player_play = pd.read_csv(f'{directory}/player_play.csv')
plays = pd.read_csv(f'{directory}/plays.csv')
tracking1 = pd.read_csv(f'{directory}/tracking_week_1.csv')
tracking2 = pd.read_csv(f'{directory}/tracking_week_2.csv')
tracking3 = pd.read_csv(f'{directory}/tracking_week_3.csv')
tracking4 = pd.read_csv(f'{directory}/tracking_week_4.csv')
tracking5 = pd.read_csv(f'{directory}/tracking_week_5.csv')
tracking6 = pd.read_csv(f'{directory}/tracking_week_6.csv')
tracking7 = pd.read_csv(f'{directory}/tracking_week_7.csv')
tracking8 = pd.read_csv(f'{directory}/tracking_week_8.csv')
tracking9 = pd.read_csv(f'{directory}/tracking_week_9.csv')

tracking_files = [tracking1, tracking2, tracking3, tracking4, tracking5,
                  tracking6, tracking7, tracking8, tracking9]

## identify TE in motion ##
# merging to get player position in player plays
player_play = pd.merge(
    player_play, 
    players[['nflId', 'position', 'displayName']],
    on='nflId', 
    how='left'
)

# filter for TE motion within player_play
te_motion = player_play[(player_play['position'] == 'TE') & 
                        (player_play['motionSinceLineset'] == True)]

# extract gameId and playId combos for these TE motion plays
te_motion_plays = te_motion[['gameId', 'playId']].drop_duplicates()

#print(te_motion_plays)

# combine all tracking csv files together 
tracking = pd.concat([tracking1,tracking2, tracking3, tracking4, tracking5,
                  tracking6, tracking7, tracking8, tracking9], axis=0)

tracking_te_motion = pd.merge(
    tracking, 
    te_motion_plays, 
    on=['gameId', 'playId'], 
    how='inner'
)

#print(len(tracking_te_motion))
#print(tracking_te_motion.shape)
#print(tracking_te_motion.head())
#print(tracking_te_motion.columns)

# merge everything into the tracking df from player_play that's needed
tracking_te_motion = pd.merge(
    tracking_te_motion,
    player_play[['gameId', 'playId', 'nflId', 'position', 'displayName',
                 'teamAbbr', 'hadRushAttempt', 'rushingYards', 'inMotionAtBallSnap', 
                 'shiftSinceLineset', 'motionSinceLineset', 'wasRunningRoute', 
                 'routeRan', 'pff_defensiveCoverageAssignment',
                 'pff_primaryDefensiveCoverageMatchupNflId', 
                 'pff_secondaryDefensiveCoverageMatchupNflId'
                 ]],
    on=['gameId', 'playId', 'nflId'],
    how='left'
)

# add play info into df
tracking_te_motion = pd.merge(
       tracking_te_motion,
       plays[['gameId', 'playId', 'possessionTeam', 'defensiveTeam', 'pff_manZone',
              'playNullifiedByPenalty', 'expectedPoints', 'offenseFormation',
              'receiverAlignment', 'rushLocationType', 'yardsGained', 
              'expectedPointsAdded', 'pff_runConceptPrimary', 'pff_runConceptSecondary',
              'pff_runPassOption', 'pff_passCoverage'
           ]],
       on=['gameId', 'playId'],
       how='left'
)

print(tracking_te_motion.shape)

# filter out plays w penalties
tracking_te_motion = tracking_te_motion[tracking_te_motion['playNullifiedByPenalty'] != 'Y']

# identifying run/rush plays
run_play_ids = player_play[player_play['hadRushAttempt'] == 1][['gameId', 'playId']]

# identify zone defense plays
zone_play_ids = plays[plays['pff_manZone'] == 'Zone'][['gameId', 'playId']]

# merge run and zone
zone_run_plays = pd.merge(
    zone_play_ids,
    run_play_ids,
    on=['gameId', 'playId'],
    how='inner'
)

tracking_te_motion = pd.merge(
    tracking_te_motion,
    zone_run_plays[['gameId', 'playId']],
    on=['gameId', 'playId'],
    how='inner'
)    

print(tracking_te_motion.shape)

# fix pff_runConceptSecondary column
tracking_te_motion['pff_runConceptSecondary'] = tracking_te_motion['pff_runConceptSecondary'].fillna('None').astype(str)

## frame filtering for pre-motion, post-motion, one-second-after-snap, and post-play
## pre motion
premotion_rows_list = []
for (gameId, playId, nflId), play_data in tracking_te_motion.groupby(['gameId', 'playId', 'nflId']):
    premotion_row = play_data[play_data['event'] == 'man_in_motion']
    
    incorrect_row = premotion_row[premotion_row['frameType'] == 'AFTER_SNAP']

    if not incorrect_row.empty:
        replacement_row = play_data[play_data['event'] == 'line_set']
        premotion_row = pd.concat([premotion_row.drop(incorrect_row.index), replacement_row])
        
    if premotion_row.empty:
        premotion_row = play_data[play_data['frameId'] == 1]
        
    premotion_rows_list.append(premotion_row)

premotion_rows = pd.concat(premotion_rows_list, axis=0)

## post motion
postmotion_rows_list = []
for (gameId, playId, nflId), play_data in tracking_te_motion.groupby(['gameId', 'playId', 'nflId']):
    postmotion_row = play_data[play_data['event'] == 'ball_snap']
    if postmotion_row.empty:
        postmotion_row = play_data[play_data['frameType'] == 'SNAP']
    postmotion_rows_list.append(postmotion_row)

postmotion_rows = pd.concat(postmotion_rows_list, axis=0)

## one second after snap
one_sec_after_rows_list = []
for (gameId, playId, nflId), play_data in tracking_te_motion.groupby(['gameId', 'playId', 'nflId']):
    # identify snap frame
    snap_frame = play_data[play_data['frameType'] == 'SNAP']
    if not snap_frame.empty:
        snap_frame_id = snap_frame['frameId'].iloc[0]
        one_sec_after_frame_id = snap_frame_id + 10 # frames = 0.1 sec each
        one_sec_after_row = play_data[play_data['frameId'] == one_sec_after_frame_id]
        one_sec_after_rows_list.append(one_sec_after_row)
        
one_sec_after_rows = pd.concat(one_sec_after_rows_list, axis=0)
# creating new 'ONE_SEC' frameType for easier classification
one_sec_after_rows['frameType'] = 'ONE_SEC'

## post play
postplay_rows_list = []
for (gameId, playId, nflId), play_data in tracking_te_motion.groupby(['gameId', 'playId', 'nflId']):
    after_snap_df = play_data[play_data['frameType'] == 'AFTER_SNAP']
    if not after_snap_df.empty:
        postplay_row = after_snap_df[after_snap_df['frameId'] == after_snap_df['frameId'].max()]
    else:
        postplay_row = play_data[play_data['frameId'] == play_data['frameId'].max()]
    
    postplay_rows_list.append(postplay_row)

postplay_rows = pd.concat(postplay_rows_list, axis=0)

# combine rows into dataset
tracking_te_motion_filtered = pd.concat([premotion_rows, postmotion_rows, one_sec_after_rows, postplay_rows])

'''
# fixing two displayName columns issue
if 'displayName_x' in tracking_te_motion_filtered.columns and 'displayName_y' in tracking_te_motion_filtered.columns:
    tracking_te_motion_filtered['displayName'] = tracking_te_motion_filtered['displayName_x'].combine_first(tracking_te_motion_filtered['displayName_y'])
    tracking_te_motion_filtered.drop(['displayName_x', 'displayName_y'], axis=1, inplace=True)
'''    
    
## adding the football back in
# extracting ball data
ball_data = tracking[tracking['displayName'] == 'football']

# filter for relevant plays (gameId and playId combos froms te data)
relevant_plays = set(zip(tracking_te_motion_filtered['gameId'], tracking_te_motion_filtered['playId']))
ball_data = ball_data[ball_data.apply(lambda row: (row['gameId'], row['playId']) in relevant_plays, axis=1)]

# frame filtering for pre-motion, post-motion, and post-play
# premotion
ball_premotion_rows_list = []

for (gameId, playId), play_data in ball_data.groupby(['gameId', 'playId']):
    ball_premotion_row = play_data[play_data['event'] == 'man_in_motion']
    
    if ball_premotion_row.empty or ball_premotion_row.iloc[0]['frameType'] == 'AFTER_SNAP':
        ball_premotion_row = play_data[play_data['event'] == 'line_set']
        
    if ball_premotion_row.empty:
        ball_premotion_row = play_data[play_data['frameId'] == 1]
        
    ball_premotion_rows_list.append(ball_premotion_row)
        
ball_premotion_rows = pd.concat(ball_premotion_rows_list, axis=0) if ball_premotion_rows_list else pd.DataFrame()

# postmotion
ball_postmotion_rows_list = []
for (gameId, playId), play_data in ball_data.groupby(['gameId', 'playId']):
    ball_postmotion_row = play_data[play_data['event'] == 'ball_snap']
    
    if ball_postmotion_row.empty:
        ball_postmotion_row = play_data[play_data['frameType'] == 'SNAP']
    
    ball_postmotion_rows_list.append(ball_postmotion_row)

ball_postmotion_rows = pd.concat(ball_postmotion_rows_list, axis=0) if ball_postmotion_rows_list else pd.DataFrame()

# one sec after snap
ball_one_sec_after_rows_list = []

for (gameId, playId), play_data in ball_data.groupby(['gameId', 'playId']):
    ball_snap_row = play_data[play_data['frameType'] == 'SNAP']
    
    if not ball_snap_row.empty:
        snap_frame_id = ball_snap_row.iloc[0]['frameId']
        one_sec_after_frame_id = snap_frame_id + 10
        
        ball_one_sec_after_row = play_data[play_data['frameId'] == one_sec_after_frame_id]
        if not ball_one_sec_after_row.empty:
            ball_one_sec_after_rows_list.append(ball_one_sec_after_row)
            
ball_one_sec_after_rows = pd.concat(ball_one_sec_after_rows_list, axis=0) if ball_one_sec_after_rows_list else pd.DataFrame()
ball_one_sec_after_rows['frameType'] = 'ONE_SEC'

# postplay
ball_postplay_rows_list = []
for (gameId, playId), play_data in ball_data.groupby(['gameId', 'playId']):
    ball_after_snap_df = play_data[play_data['frameType'] == 'AFTER_SNAP']
   
    if not ball_after_snap_df.empty:
        ball_postplay_row = ball_after_snap_df[ball_after_snap_df['frameId'] == ball_after_snap_df['frameId'].max()]
   
    else:
        ball_postplay_row = play_data[play_data['frameId'] == play_data['frameId'].max()]
        
    ball_postplay_rows_list.append(ball_postplay_row)
    
ball_postplay_rows = pd.concat(ball_postplay_rows_list, axis=0) if ball_postplay_rows_list else pd.DataFrame()

# combine all fb rows
all_ball_rows = pd.concat([ball_premotion_rows, ball_postmotion_rows, ball_one_sec_after_rows, ball_postplay_rows], axis=0)

# put back in big dataset
tracking_te_motion_filtered = pd.concat([tracking_te_motion_filtered, all_ball_rows], axis=0)

# sort data 
tracking_te_motion_filtered = tracking_te_motion_filtered.sort_values(by=['gameId', 'playId', 'nflId'])

tracking_te_motion_filtered.rename(columns={'displayName_x': 'displayName'}, inplace=True)

# fixing displayName missing for football issue
tracking_te_motion_filtered['displayName'] = tracking_te_motion_filtered.apply(
    lambda row: 'football' if row['club'] == 'football' else row['displayName'], axis=1
)

tracking_te_motion_filtered.to_csv('te_run_zone_data_fb.csv', index=False)

'''
test_df = tracking_te_motion_filtered[
    (tracking_te_motion_filtered['gameId'] == 2022090800) &
    (tracking_te_motion_filtered['playId'] == 2043) &
    (tracking_te_motion_filtered['nflId'] == 35472) 
    ]

print(test_df)
print(test_df.shape)
'''