#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

data = pd.read_csv('unified_data.csv')

data = data.dropna(subset=['absolute_change_area_snap', 'binary_success'])
data

'''
using point-biserial correlations since success metric is binary!!
'''

# compute correlation with success
#corr_absolute = data['absolute_change_area_snap'].corr(data['binary_success'])
#corr_percentage = data['percentage_change_area_snap'].corr(data['binary_success'])

corr_ab, p_value = pointbiserialr(data['absolute_change_area_snap'], data['binary_success'])
corr_pc, p_value = pointbiserialr(data['percentage_change_area_snap'], data['binary_success'])

print(f'Correlation between absolute space change and success: {corr_ab}')
print(f'Correlation between percentage space change and success: {corr_pc}')

te_motion_zone = pd.read_csv('te_defender_spacing_change_zone.csv')
# compute correlation with success
#corr_absolute_zone = te_motion_zone['absolute_change_area_snap'].corr(te_motion_zone['binary_success'])
#corr_percentage_zone = te_motion_zone['percentage_change_area_snap'].corr(te_motion_zone['binary_success'])

te_zone_corr_ab, p_value = pointbiserialr(te_motion_zone['absolute_change_area_snap'], te_motion_zone['binary_success'])
te_zone_corr_pc, p_value = pointbiserialr(te_motion_zone['percentage_change_area_snap'], te_motion_zone['binary_success'])

print(f'Correlation between absolute space change and success (zone): {te_zone_corr_ab}')
print(f'Correlation between percentage space change and success (zone): {te_zone_corr_pc}')

te_motion_man = pd.read_csv('te_defender_spacing_change_man.csv')
# compute correlation with success
#corr_absolute_man = te_motion_man['absolute_change_area_snap'].corr(te_motion_man['binary_success'])
#corr_percentage_man = te_motion_man['percentage_change_area_snap'].corr(te_motion_man['binary_success'])

te_man_corr_ab, p_value = pointbiserialr(te_motion_man['absolute_change_area_snap'], te_motion_man['binary_success'])
te_man_corr_pc, p_value = pointbiserialr(te_motion_man['percentage_change_area_snap'], te_motion_man['binary_success'])

print(f'Correlation between absolute space change and success (man): {te_man_corr_ab}')
print(f'Correlation between percentage space change and success (man): {te_man_corr_pc}')

no_motion_man = pd.read_csv('no_motion_man_spacing.csv')
# compute correlation with success
#corr_absolute_man_nm = no_motion_man['absolute_change_area_snap'].corr(no_motion_man['binary_success'])
#corr_percentage_man_nm = no_motion_man['percentage_change_area_snap'].corr(no_motion_man['binary_success'])

nm_man_corr_ab, p_value = pointbiserialr(no_motion_man['absolute_change_area_snap'], no_motion_man['binary_success'])
nm_man_corr_pc, p_value = pointbiserialr(no_motion_man['percentage_change_area_snap'], no_motion_man['binary_success'])

print(f'Correlation between absolute space change and success (man nm): {nm_man_corr_ab}')
print(f'Correlation between percentage space change and success (man nm): {nm_man_corr_pc}')

no_motion_zone = pd.read_csv('no_motion_zone_spacing.csv')
no_motion_zone = no_motion_zone.dropna(subset=['absolute_change_area_snap', 'binary_success'])
# compute correlation with success
#corr_absolute_zone_nm = no_motion_zone['absolute_change_area_snap'].corr(no_motion_zone['binary_success'])
#corr_percentage_zone_nm = no_motion_zone['percentage_change_area_snap'].corr(no_motion_zone['binary_success'])

nm_zone_corr_ab, p_value = pointbiserialr(no_motion_zone['absolute_change_area_snap'], no_motion_zone['binary_success'])
nm_zone_corr_pc, p_value = pointbiserialr(no_motion_zone['percentage_change_area_snap'], no_motion_zone['binary_success'])

print(f'Correlation between absolute space change and success (zone nm): {nm_zone_corr_ab}')
print(f'Correlation between percentage space change and success (zone nm): {nm_zone_corr_pc}')
