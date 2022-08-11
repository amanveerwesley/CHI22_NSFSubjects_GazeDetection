# OpenVokaWavMean-win64.py
# public-domain sample code by Vokaturi, 2019-05-31
#
# A sample script that uses the VokaturiPlus library to extract the emotions from
# a wav file on disk. The file has to contain a mono recording.
#
# Call syntax:
#   python3 OpenVokaWavMean-win64.py path_to_sound_file.wav
#
# For the sound file hello.wav that comes with OpenVokaturi, the result should be:
#	Neutral: 0.760
#	Happy: 0.000
#	Sad: 0.238
#	Angry: 0.001
#	Fear: 0.000

# Explanation of audio analysis available at https://developers.vokaturi.com/algorithms/acoustic-features
# Code modified for data collection by Christopher Blank

from os import listdir
import pandas as pd

import sys
import scipy.io.wavfile

sys.path.append("../api")
import Vokaturi

print ("Loading library...")
Vokaturi.load("../lib/open/win/OpenVokaturi-3-3-win64.dll")
print ("Analyzed by: %s" % Vokaturi.versionAndLicense())


# Get all wav files from directory
def find_wav_filenames( path_to_dir, suffix=".wav" ):
	filenames = listdir(path_to_dir)
	return [ filename for filename in filenames if filename.endswith( suffix ) ]

# wav files from subject facial videos
filenames = find_wav_filenames("E:/NSF-Email_FacialVideos_63Subjects/Audio")

for name in filenames:
	print(name[:4])
	new_csv = name[:4] + '_Sound.csv'

	print ("Reading sound file for: "+name[:4])
	file_name = "E:/NSF-Email_FacialVideos_63Subjects/Audio/" + name
	(sample_rate, samples) = scipy.io.wavfile.read(file_name)
	print ("   sample rate %.3f Hz" % sample_rate)
	
	print ("Allocating Vokaturi sample array...")
	buffer_length = len(samples) 
	print ("   %d samples, %d channels" % (buffer_length, samples.ndim))
	
	seconds = int(buffer_length / sample_rate)
	
	#
	second_list = []
	#
	second_with_data_list = []
	neutral_list = []
	happy_list = []
	sad_list = []
	angry_list = []
	afraid_list = []

	for i in range(seconds): #go till end of file
		
		second_list.append(float(i))
		
		# feed each second of data into the algorithm
		if i == 0:
			new_buffer_start = sample_rate
		else:
			new_buffer_start = i*sample_rate
			
		new_buffer_end = new_buffer_start + sample_rate
		
		sub_samples = samples[new_buffer_start : new_buffer_end]
		
		c_buffer = Vokaturi.SampleArrayC(sample_rate)
		if samples.ndim == 1:  # mono
			c_buffer[:] = sub_samples[:] / 32768.0
		else:  # stereo
			c_buffer[:] = 0.5*(sub_samples[:,0]+0.0+sub_samples[:,1]) / 32768.0
		
		# Creating VokaturiVoice
		voice = Vokaturi.Voice (sample_rate, len(sub_samples))
		
		# Filling VokaturiVoice with samples
		voice.fill(sample_rate, c_buffer) # It would be better to loop from this point if possible. I don't know how, though.
		
		# Extracting emotions from VokaturiVoice
		quality = Vokaturi.Quality()
		emotionProbabilities = Vokaturi.EmotionProbabilities()
		voice.extract(quality, emotionProbabilities)
		
		if quality.valid:
			second_with_data_list.append(float(i))
			print ("Neutral: %.3f" % emotionProbabilities.neutrality)
			neutral_list.append(emotionProbabilities.neutrality)
			print ("Happy: %.3f" % emotionProbabilities.happiness)
			happy_list.append(emotionProbabilities.happiness)
			print ("Sad: %.3f" % emotionProbabilities.sadness)
			sad_list.append(emotionProbabilities.sadness)
			print ("Angry: %.3f" % emotionProbabilities.anger)
			angry_list.append(emotionProbabilities.anger)
			print ("Fear: %.3f" % emotionProbabilities.fear)
			afraid_list.append(emotionProbabilities.fear)
		else:
			print ("Not enough sonorancy to determine emotions")
		
		print(str(i) + ' of ' + str(seconds) + ' seconds processed')
		
		voice.destroy()
	# Append new_predictions to CSV file
	Sound_df = pd.DataFrame(list(zip(second_with_data_list,angry_list,afraid_list,happy_list,sad_list,neutral_list)),columns=['Seconds','S_Angry', 'S_Afraid', 'S_Happy', 'S_Sad',  'S_Neutral'])
	
	Second_df = pd.DataFrame(second_list, columns = ['Seconds'])
	
	Sound_df = pd.merge_ordered(Sound_df,Second_df,on='Seconds',how='outer')
	
	Sound_df.insert(0, 'Participant_ID', name[:4])
	
	Sound_df.to_csv(new_csv,index=False)
