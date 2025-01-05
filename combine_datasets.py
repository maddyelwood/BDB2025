#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# load in all 4 csv files
te_motion_zone = pd.read_csv('te_defender_spacing_change_zone.csv')
te_motion_man = pd.read_csv('te_defender_spacing_change_man.csv')
no_motion_zone = pd.read_csv('no_motion_zone_spacing.csv')
no_motion_man = pd.read_csv('no_motion_man_spacing.csv')

# add binary flags for features
# covereageType: zone = 0, man = 1
# motion: none = 0, motion = 1
te_motion_zone['coverageType'] = 0
te_motion_zone['motion'] = 1

te_motion_man['coverageType'] = 1
te_motion_man['motion'] = 1

no_motion_zone['coverageType'] = 0
no_motion_zone['motion'] = 0

no_motion_man['coverageType'] = 1
no_motion_man['motion'] = 0

# combine datasets
unified_data = pd.concat([te_motion_zone, te_motion_man, no_motion_zone, no_motion_man], ignore_index=True)

# save to new csv file for future use
unified_data.to_csv('unified_data.csv', index=False)
