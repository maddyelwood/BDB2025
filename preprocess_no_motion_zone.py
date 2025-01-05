#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

directory = 'nfl_data'

# Load the datasets
players = pd.read_csv(f'{directory}/players.csv')
player_play = pd.read_csv(f'{directory}/player_play.csv')
plays = pd.read_csv(f'{directory}/plays.csv')

# optimize memory usage
players['nflId'] = players['nflId'].astype('int32')
player_play['nflId'] = player_play['nflId'].astype('int32')
player_play['gameId'] = player_play['gameId'].astype('int32')
player_play['playId'] = player_play['playId'].astype('int32')
plays['gameId'] = plays['gameId'].astype('int32')
plays['playId'] = plays['playId'].astype('int32')

print(f'players: {players.shape}')
print(f'player-play: {player_play.shape}')
print(f'plays: {plays.shape}')


# Merge player positions into player_play
player_play = pd.merge(
    player_play, 
    players[['nflId', 'position', 'displayName']],
    on='nflId', 
    how='left'
)


# function to process each tracking file in chunks and extract 4 frames
def process_tracking_file_frames(file_path, no_motion_play_ids):
    chunk_size = 1_000_000 
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    
    processed_chunks = []
    for chunk in chunks:
        # filter chunk for no-motion plays
        filtered_chunk = pd.merge(
            chunk,
            no_motion_play_ids,
            on=['gameId', 'playId'],
            how='inner'
        )
    
        # extract frames
        frames = []
        print(f'processing chunk: {len(frames)} frames for file {file_path}')
        
        for (game_id, play_id), group in filtered_chunk.groupby(['gameId', 'playId']):
            print(f"processing gameId: {game_id}, playId: {play_id}, players: {len(group['nflId'].unique())}")
            
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
            frames = []
        
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

print(f'no motion: {no_motion_play_ids.shape}')

# Filter tracking data for these no-motion plays
tracking_no_motion = pd.DataFrame()

# process each tracking file in chunks
tracking_files = [f'{directory}/tracking_week_{i}.csv' for i in range(1, 10)]
for file_path in tracking_files:
    tracking_no_motion = pd.concat(
        [tracking_no_motion, process_tracking_file_frames(file_path, no_motion_play_ids)],
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
zone_play_ids = plays[plays['pff_manZone'] == 'Zone'][['gameId', 'playId']]

# Merge run plays and zone plays
zone_run_plays = pd.merge(
    zone_play_ids,
    run_play_ids,
    on=['gameId', 'playId'],
    how='inner'
)

# Keep only run plays under zone defense
print('testing zone run merge...')
tracking_no_motion = pd.merge(
    tracking_no_motion,
    zone_run_plays[['gameId', 'playId']],
    on=['gameId', 'playId'],
    how='inner'
)
print('zone run merge successful')

# Fix pff_runConceptSecondary column
tracking_no_motion['pff_runConceptSecondary'] = tracking_no_motion['pff_runConceptSecondary'].fillna('None').astype(str)

print('testing position merge...')
tracking_no_motion = pd.merge(
    tracking_no_motion,
    players[['nflId', 'position']],
    on='nflId',
    how='left'
)
print('position merge successful')

'''
going to deal with this in lane detect program instead

print('testing chunked teamAbbr merge...')
tracking_no_motion_players = tracking_no_motion[tracking_no_motion['nflId'].notna()]

team_abbr_chunks = []
chunk_size = 100_000
for chunk in pd.read_csv(f'{directory}/player_play.csv', chunksize=chunk_size):
    chunk = chunk[['nflId', 'teamAbbr']]
    merged_chunk = pd.merge(tracking_no_motion_players, chunk, on='nflId', how='left')
    team_abbr_chunks.append(merged_chunk)
tracking_no_motion_players = pd.concat(team_abbr_chunks, ignore_index=True)

print('before adding football rows')
football_rows = tracking_no_motion[tracking_no_motion['nflId'].isna()]
print('after adding football rows')

tracking_no_motion = pd.concat([tracking_no_motion_players, football_rows], ignore_index=True)

print('chunked teamAbbr merge successful')
'''
'''
tracking_no_motion = pd.merge(
    tracking_no_motion,
    player_play[['nflId', 'teamAbbr']],
    on='nflId',
    how='left'
)
'''

print(tracking_no_motion[['nflId', 'position']].head())

print(tracking_no_motion.shape)

tracking_no_motion.to_csv('preprocessed_no_motion_zone.csv', index=False)
