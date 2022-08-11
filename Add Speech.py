# -*- coding: utf-8 -*-
"""
Created on Fri August 8 1:30:58 2019

@author: Christopher Blank
"""

import pandas as pd
from os import listdir

# Get all csv files from directory
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# Path to speech csvs
S_filenames = find_csv_filenames("C:/Users/despe/source/repos/UH CS REU 2019/Summer/OpenVokaturi-3-3/OpenVokaturi-3-3/examples/Le Source")

# List of files with sound
read = []

for name in S_filenames:
	# FACS+Gaze+Session csvs
	F_filename = name[:4]
	read.append(F_filename)
	
	print(F_filename)
	
	# Access csvs
	F_df = pd.read_csv("E:/NSF-Email_FacialVideos_63Subjects/Results/" + F_filename + "_FACS+Gaze.csv")
	S_df = pd.read_csv("C:/Users/despe/source/repos/UH CS REU 2019/Summer/OpenVokaturi-3-3/OpenVokaturi-3-3/examples/Le Source/" + name)
	
	S_df.rename(columns = {'Participant':'Participant_ID'}, inplace=True)
	
	FS_df = pd.merge_ordered(F_df,S_df,how='outer')
	
	FS_df.to_csv(F_filename+'_FACS+Gaze+Speech Data.csv',index=False)

print('')

# Repeat to enter empty columns for the six soundless participants
F_filenames = find_csv_filenames("E:/NSF-Email_FacialVideos_63Subjects/Results")

for name in F_filenames:
	F_filename = name[:4]
	if F_filename not in read:
		print(name[:4])
		F_df = pd.read_csv("E:/NSF-Email_FacialVideos_63Subjects/Results/" + name)

		# Enter empty columns for the six soundless participants
		F_df = F_df.reindex(columns = F_df.columns.tolist() + ['S_Angry','S_Afraid','S_Happy','S_Sad','S_Neutral'])
		
		F_df.to_csv(F_filename+'_FACS+Gaze+Speech Data.csv',index=False)