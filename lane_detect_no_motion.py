#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

#tracking_data = pd.read_csv('preprocessed_no_motion_zone.csv')
tracking_data = pd.read_csv('preprocessed_no_motion_man.csv')

player_play = pd.read_csv('nfl_data/player_play.csv')
team_abbr_map = player_play.set_index('nflId')['teamAbbr'].to_dict()

def get_team_abbr(nflId):
    return team_abbr_map.get(nflId, None)

# Calculate the area of the polygon formed by defenders
def calc_area(defenders):
    if len(defenders) < 3:
        return np.nan
    
    x_coords = defenders['x'].values
    y_coords = defenders['y'].values
    
    x_coords = np.append(x_coords, x_coords[0])
    y_coords = np.append(y_coords, y_coords[0])
    
    area = 0.5 * np.abs(
        np.dot(x_coords[:-1], y_coords[1:]) - np.dot(y_coords[:-1], x_coords[1:])
    )
    return area

def filter_runs_matching_te_side(df):
    def determine_te_side(row, ball_y):
        if row['playDirection'] == 'right':
            return 'RIGHT' if row['y'] < ball_y else 'LEFT'
    
        else:
            return 'RIGHT' if row['y'] > ball_y else 'LEFT'
    
    # extract snap rows for TEs only
    snap_tes = df[(df['position'] == 'TE') & (df['frameType'] == 'SNAP')].copy()
    
    # extract ball position at snap
    snap_ball = df[(df['frameType'] == 'SNAP') & (df['club'] == 'football')].copy()
    snap_ball = snap_ball.rename(columns={'y': 'ball_y'})[['gameId', 'playId', 'ball_y']]
    
    # merge ball position into te data
    snap_tes = pd.merge(snap_tes, snap_ball, on=['gameId', 'playId'], how='left')
    
    # determine te side using ball_y column
    snap_tes['teSide'] = snap_tes.apply(lambda row: determine_te_side(row, row['ball_y']), axis=1)
    
    # merge te side info back into the main dataframe
    df = pd.merge(
        df,
        snap_tes[['gameId', 'playId', 'teSide']].drop_duplicates(),
        on=['gameId', 'playId'],
        how='left'
    )
    
    # match rushLocationType with TE side
    def match_rush_to_te_side(row):
        # check is rushLocationType matches TE side
        rush_side = 'LEFT' if 'LEFT' in str(row['rushLocationType']) else 'RIGHT'
        return rush_side == row['teSide']
    
    filtered_df = df[df.apply(match_rush_to_te_side, axis=1)]
    
    return filtered_df

print(f'Unfiltered dataset shape: {tracking_data.shape}')

# apply filtering based on TE premotion side
filtered_df = filter_runs_matching_te_side(tracking_data)
print(f'Filtered dataset shape: {filtered_df.shape}')
        

# results df
spacing_results = []

# Group by play
for (gameId, playId), play_data in filtered_df.groupby(['gameId', 'playId']):
    play_result = {'gameId': gameId, 'playId': playId}
    
    # Identify defenders from snap frame
    snap_data = play_data[play_data['frameType'] == 'SNAP']
    te_data = snap_data[snap_data['position'] == 'TE']
    
    if te_data.empty:
        continue
    
    # add teamAbbr column to snap data
    if 'teamAbbr' not in snap_data.columns:
        snap_data.loc[:,'teamAbbr'] = snap_data['nflId'].apply(get_team_abbr)
    
    te_row = te_data.iloc[0]
    te_team_abbr = get_team_abbr(te_row['nflId'])
    te_data.loc[te_data.index[0],'teamAbbr'] = te_team_abbr
    defender_data = snap_data[snap_data['teamAbbr'] != te_team_abbr]
    
    if defender_data.empty:
        continue
    
    # Count defensive players by position
    linemen_count = defender_data[defender_data['position'].isin(['DE', 'DT', 'NT'])].shape[0]
    linebackers_count = defender_data[defender_data['position'].isin(['OLB', 'MLB', 'ILB', 'LB'])].shape[0]
    
    play_result['linemen_count'] = linemen_count
    play_result['linebackers_count'] = linebackers_count
    
    if linemen_count < 3:
        continue

    linebacker_selection_count = 1 if linebackers_count < 3 else 2
    
    # Calculate distance to TE for all defenders
    defender_data = defender_data.copy()
    defender_data['distance_to_te'] = np.sqrt(
        (defender_data['x'] - te_row['x'])**2 + (defender_data['y'] - te_row['y'])**2
    )
    
    # Select defenders closest to the TE
    try:
        selected_defenders = pd.concat([
            defender_data[defender_data['position'] == 'DE'].nsmallest(1, 'distance_to_te'),
            defender_data[defender_data['position'].isin(['DT', 'NT'])].nsmallest(1, 'distance_to_te'),
            defender_data[defender_data['position'].isin(['OLB', 'MLB', 'ILB'])].nsmallest(linebacker_selection_count, 'distance_to_te')
        ])
    except KeyError:
        continue
    
    if selected_defenders.shape[0] < (2 + linebacker_selection_count):
        continue
    
    snap_defenders = selected_defenders['nflId'].tolist()
    
    
    # Process each frame type
    for frameType in ['BEFORE_SNAP', 'SNAP', 'ONE_SEC', 'AFTER_SNAP']:
        frame_data = play_data[play_data['frameType'] == frameType]
        
        # put teamAbbr column in for frame_data
        if 'teamAbbr' not in frame_data.columns:
            frame_data = frame_data.copy()
            frame_data['teamAbbr'] = frame_data['nflId'].apply(get_team_abbr)
        
        if 'teamAbbr' in frame_data.columns:
            #print('teamAbbr exists in frame_data')
            
            if 'teamAbbr' not in te_row:
                te_nfl_id = te_row['nflId']
                te_team_abbr = get_team_abbr(te_nfl_id)
                te_row['teamAbbr'] = te_team_abbr
            
            defender_data = frame_data[frame_data['teamAbbr'] != te_row['teamAbbr']]  
            
        else:
            print('teamAbbr column not found in frame_data')
            
        consistent_defenders = defender_data[defender_data['nflId'].isin(snap_defenders)]
        
        if consistent_defenders.shape[0] < len(snap_defenders):
            continue
        
        area = calc_area(consistent_defenders)
        play_result[f'area_polygon_{frameType}'] = area
        
        try:
            de_row = selected_defenders[selected_defenders['position'] == 'DE'].iloc[0]
            dt_row = selected_defenders[selected_defenders['position'].isin(['DT', 'NT'])].iloc[0]
            
            distance_de_dt = np.sqrt((de_row['x'] - dt_row['x'])**2 + (de_row['y'] - dt_row['y'])**2)
            play_result[f'width_of_lane_front_{frameType}'] = distance_de_dt
        except IndexError:
            play_result[f'width_of_lane_front_{frameType}'] = np.nan
    
    # calc changes in area
    before_snap_area = play_result.get('area_polygon_BEFORE_SNAP', np.nan)
    snap_area = play_result.get('area_polygon_SNAP', np.nan)
    one_sec_area = play_result.get('area_polygon_ONE_SEC', np.nan)
    
    # between before snap and at snap
    if not np.isnan(before_snap_area) and not np.isnan(snap_area):
        # absolute change
        play_result['absolute_change_area_snap'] = snap_area - before_snap_area
        
        # percentage change
        play_result['percentage_change_area_snap'] = (
            ((snap_area - before_snap_area) / before_snap_area) * 100
            if before_snap_area != 0 else np.nan
        )
    else:
        play_result['absolute_change_area_snap'] = np.nan
        play_result['percentage_change_area_snap'] = np.nan
       
    # between snap and one sec later
    if not np.isnan(snap_area) and not np.isnan(one_sec_area):
        # absolute change
        play_result['absolute_change_area_one_sec'] = one_sec_area - snap_area
        
        # percentage change
        play_result['percentage_change_area_one_sec'] = (
            ((one_sec_area - snap_area) / snap_area) * 100
            if snap_area != 0 else np.nan
        )
    else:
        play_result['absolute_change_area_one_sec'] = np.nan
        play_result['percentage_change_area_one_sec'] = np.nan
    
    # calc changes in width_of_lane_front
    width_before_snap = play_result.get('width_of_lane_front_BEFORE_SNAP', np.nan)
    #print(f'width_before_snap: {width_before_snap}')
    width_snap = play_result.get('width_of_lane_front_SNAP', np.nan)
    #print(f'width_snap: {width_snap}')
    width_one_sec = play_result.get('width_of_lane_front_ONE_SEC', np.nan)
    
    # between before snap and snap
    if not np.isnan(width_before_snap) and not np.isnan(width_snap):
        #print(f'calculating changes for playId {playId}: BEFORE_SNAP={width_before_snap}, SNAP={width_snap}')
        
       # absolute change
        play_result['absolute_change_width_snap'] = width_snap - width_before_snap
        
        # percentage change
        play_result['percentage_change_width_snap'] = (
            ((width_snap - width_before_snap) / width_before_snap) * 100
            if width_before_snap != 0 else np.nan
        )
    else:
        play_result['absolute_change_width_snap'] = np.nan
        play_result['percentage_change_width_snap'] = np.nan
    
    # between snap and one sec later
    if not np.isnan(width_snap) and not np.isnan(width_one_sec):
        #print(f'calculating changes for playId {playId}: BEFORE_SNAP={width_before_snap}, SNAP={width_snap}')
        
       # absolute change
        play_result['absolute_change_width_one_sec'] = width_one_sec - width_snap
        
        # percentage change
        play_result['percentage_change_width_one_sec'] = (
            ((width_one_sec - width_snap) / width_snap) * 100
            if width_snap != 0 else np.nan
        )
    else:
        play_result['absolute_change_width_one_sec'] = np.nan
        play_result['percentage_change_width_one_sec'] = np.nan
    

    spacing_results.append(play_result)

    # add in additional needed info from original dataset
    try:
        play_info = tracking_data[(tracking_data['gameId'] == gameId) & 
                                  (tracking_data['playId'] == playId)].iloc[0]
        # add EPA and yardsGained columns
        play_result['expectedPointsAdded'] = play_info['expectedPointsAdded']
        play_result['yardsGained'] = play_info['yardsGained']
     
        # add success rate based on EPA
        play_result['binary_success'] = 1 if play_info['expectedPointsAdded'] > 0 else 0
     
    except IndexError:
        play_result['expectedPointsAdded'] = np.nan
        play_result['yardsGained'] = np.nan
        play_result['binary_success'] = np.nan

# Create a DataFrame for results
results_df = pd.DataFrame(spacing_results)

# Save results to CSV
#results_df.to_csv('no_motion_zone_spacing.csv', index=False)
results_df.to_csv('no_motion_man_spacing.csv', index=False)

