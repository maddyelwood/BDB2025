#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

directory = 'nfl_data'

# Load the datasets
players = pd.read_csv(f'{directory}/players.csv')
player_play = pd.read_csv(f'{directory}/player_play.csv')
plays = pd.read_csv(f'{directory}/plays.csv')

# Merge player positions into player_play
player_play = pd.merge(
    player_play, 
    players[['nflId', 'position', 'displayName']],
    on='nflId', 
    how='left'
)

# function to process each tracking file in chunks
def process_tracking_file(file_path, no_motion_play_ids):
    chunk_size = 1_000_000 
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    processed_chunks = []
    
    for chunk in chunks:
        filtered_chunk = pd.merge(
            chunk,
            no_motion_play_ids,
            on=['gameId', 'playId'],
            how='inner'
        )
       
        # extract frames
        frames = []
        for (game_id, play_id), group in filtered_chunk.groupby(['gameId', 'playId']):
            # loop over each player in the play
            for nflId in group['nflId'].unique():
                player_data = group[group['nflId'] == nflId]
                
                before_snap = player_data[player_data['event'] == 'line_set'].iloc[-1:] if 'line_set' in player_data['event'].values else None
                snap = player_data[player_data['frameType'] == 'SNAP'].iloc[-1:] if 'SNAP' in player_data['frameType'].values else None
                if snap is not None:
                    snap_frame_id = snap['frameId'].iloc[0]
                    one_sec_id = snap_frame_id + 10 # frames = 0.1 sec each
                    one_sec = player_data[player_data['frameId'] >= one_sec_id].iloc[:1]
                    if not one_sec.empty:
                        one_sec['frameType'] = 'ONE_SEC'
                        
                after_snap = group[group['frameType'] == 'AFTER_SNAP'].iloc[:1] if 'AFTER_SNAP' in group['frameType'].values else None
        
                # combine frames
                for frame in [before_snap, snap, one_sec, after_snap]:
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                
            # for football
            football_data = group[group['displayName'] == 'football']
           
            before_snap_football = football_data[football_data['event'] == 'line_set'].iloc[-1:] if 'line_set' in football_data['event'].values else None
            snap_football = football_data[football_data['frameType'] == 'SNAP'].iloc[-1:] if 'SNAP' in football_data['frameType'].values else None
            if snap_football is not None:
                snap_frame_id_football = snap_football['frameId'].iloc[0]
                one_sec_id_football = snap_frame_id_football + 10
                one_sec_football = football_data[football_data['frameId'] >= one_sec_id_football].iloc[:1]
                if not one_sec_football.empty:
                    one_sec_football['frameType'] = 'ONE_SEC'
            after_snap_football = football_data[football_data['frameType'] == 'AFTER_SNAP'].iloc[:1] if 'AFTER_SNAP' in football_data['frameType'].values else None
            
            for frame in [before_snap_football, snap_football, one_sec_football, after_snap_football]:
                if frame is not None and not frame.empty:
                    frames.append(frame)
            
        # combine all frames for this chunk
        if frames:
            processed_chunks.append(pd.concat(frames, axis=0))
        
    return pd.concat(processed_chunks, axis=0) if processed_chunks else pd.DataFrame()
    

# Identify plays with no motion by grouping gameId and playId to check all players
no_motion_plays = (
    player_play.groupby(['gameId', 'playId'])
    .filter(lambda group: (
        group['motionSinceLineset'].fillna(False).eq(False).all() 
        ) and (
            group['inMotionAtBallSnap'].fillna(False).eq(False).all()
            ))    
)

# Extract unique gameId and playId combinations for no-motion plays
no_motion_play_ids = no_motion_plays[['gameId', 'playId']].drop_duplicates()

# Filter tracking data for these no-motion plays
tracking_no_motion = pd.DataFrame()

# process each tracking file in chunks
tracking_files = [f'{directory}/tracking_week_{i}.csv' for i in range(1, 10)]
for file_path in tracking_files:
    tracking_no_motion = pd.concat(
        [tracking_no_motion, process_tracking_file(file_path, no_motion_play_ids)],
        axis=0
        )

# Merge play information
tracking_no_motion = pd.merge(
    tracking_no_motion,
    plays[['gameId', 'playId', 'possessionTeam', 'defensiveTeam', 'pff_manZone',
           'playNullifiedByPenalty', 'expectedPoints', 'offenseFormation',
           'receiverAlignment', 'rushLocationType', 'yardsGained', 
           'expectedPointsAdded', 'pff_runConceptPrimary', 'pff_runConceptSecondary',
           'pff_runPassOption', 'pff_passCoverage']],
    on=['gameId', 'playId'],
    how='left'
)

# Filter out plays nullified by penalties
tracking_no_motion = tracking_no_motion[tracking_no_motion['playNullifiedByPenalty'] != 'Y']

# Identify run plays
run_play_ids = player_play[player_play['hadRushAttempt'] == 1][['gameId', 'playId']]

# Identify zone defense plays
man_play_ids = plays[plays['pff_manZone'] == 'Man'][['gameId', 'playId']]

# Merge run plays and zone plays
man_run_plays = pd.merge(
    man_play_ids,
    run_play_ids,
    on=['gameId', 'playId'],
    how='inner'
)

# Keep only run plays under zone defense
tracking_no_motion = pd.merge(
    tracking_no_motion,
    man_run_plays[['gameId', 'playId']],
    on=['gameId', 'playId'],
    how='inner'
)

# Fix pff_runConceptSecondary column
tracking_no_motion['pff_runConceptSecondary'] = tracking_no_motion['pff_runConceptSecondary'].fillna('None').astype(str)

tracking_no_motion = pd.merge(
    tracking_no_motion,
    players[['nflId', 'position']],
    on='nflId',
    how='left'
)

print(tracking_no_motion.shape)

tracking_no_motion.to_csv('preprocessed_no_motion_man.csv', index=False)
