# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:54:00 2019

@author: Christopher Blank
"""

import moviepy.editor as mp
from os import listdir

# Get all csv files from directory
def find_mp4_filenames( path_to_dir, suffix=".mp4" ):
	filenames = listdir(path_to_dir)
	return [ filename for filename in filenames if filename.endswith( suffix ) ]

# Path to participant facial videos
filenames = find_mp4_filenames('E:/NSF-Email_FacialVideos_63Subjects/To Do')

# mp4 to wav
for name in filenames:
	p_id = name[7:11]
	print(p_id)
	
	#try:
	clip = mp.VideoFileClip('E:/NSF-Email_FacialVideos_63Subjects/To Do/' + name)
	clip.audio.write_audiofile(p_id+"_audio.wav")
		
	#except OSError:
	#	pass
		
