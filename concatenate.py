# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:17:46 2019

@author: Christopher Blank
"""

from os import listdir
import pandas as pd

# Get all csv files from directory
def find_csv_filenames( path_to_dir, suffix=".csv" ):
	filenames = listdir(path_to_dir)
	return [ filename for filename in filenames if filename.endswith( suffix ) ]

# subject FACS sheets
filenames = find_csv_filenames('CSV/Subjects/')

first = True

for name in filenames:
	# Data for each subject
	if(first):
		copy = pd.read_csv('CSV/Subjects/'+name)
		first = False
		
		print(name[:4])
	else:
		subject = pd.read_csv('CSV/Subjects/'+name)
	
		subject.columns = copy.columns # Prepare
		
		copy = pd.concat([copy,subject], ignore_index=True) # Concatenate
		
		print(name[:4])

print('Combining files')

copy.to_csv('FACS+Gaze+Speech Data.csv', index = False)