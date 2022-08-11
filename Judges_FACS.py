# Code for facial tracking and expression extraction taken from github.com/serengil/tensorflow-101
# Modified by Christopher Blank

# Run CSVs through Add_Timestamp.py to complete formatting
#########################################################

import numpy as np
import cv2
from keras.preprocessing import image
import pandas as pd

#-----------------------------
#opencv initialization

#face_cascade = cv2.CascadeClassifier('C:/Users/despe/source/repos/UH CS REU 2019/tensorflow-101-master/tensorflow-101-master/model/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('C:/Users/cpladmin/Desktop/FACS/lib/haarcascade_frontalface_default.xml')

# Input video file
cap = cv2.VideoCapture('C:/Users/cpladmin/Desktop/FACS/FACS-Gaze-Speech-master/Panel of Judges Video.mp4')
# Output mp4
out = cv2.VideoWriter('Judges_FACS.mp4', 25, 30.0, (int(cap.get(3)), int(cap.get(4))))

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
#model = model_from_json(open("C:/Users/despe/source/repos/UH CS REU 2019/tensorflow-101-master/tensorflow-101-master/model/facial_expression_model_structure.json", "r").read())
#model.load_weights('C:/Users/despe/source/repos/UH CS REU 2019/tensorflow-101-master/tensorflow-101-master/model/facial_expression_model_weights.h5') #load weights

model = model_from_json(open("C:/Users/cpladmin/Desktop/FACS/lib/tensorflow-101-master/model/facial_expression_model_structure.json", "r").read())
model.load_weights('C:/Users/cpladmin/Desktop/FACS/lib/tensorflow-101-master/model/facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral')

# Pseudo timekeeper
frame_counter = 0
frame_list = []
# Relative Time
time_list = []
# Judge identifyer
sector_list = []
frame_list_F = []
# Emotions
angry_list = []
disgusted_list = []
afraid_list = []
happy_list = []
sad_list = []
surprised_list = []
neutral_list = []

while(True):
	ret, img = cap.read() # read frame from file
	
	if np.shape(img) == (): # Stop reading at end of file
		break
	
	# Update frame number
	frame_counter += 1
	# Skip the video portion that records empty chairs
	if(frame_counter < 2500):
		continue
	# Store frame number
	frame_list.append(frame_counter)
	# Display frame number
	cv2.putText(img, str(frame_counter), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)	
		
	# Caltulate time
	decisecond_true = (frame_counter/3)%10
	second_total = (frame_counter/3-decisecond_true)/10
	second_true = second_total%60
	minute_total = (second_total-second_true)/60
	minute_true = minute_total%60
	hour = (minute_total-minute_true)/60
	# Format time
	decisecond_true = str(int(decisecond_true))
	if second_true < 10:
		second_true = '0' + str(int(second_true))
	else:
		second_true = str(int(second_true))
	if minute_true < 10:
		minute_true = '0' + str(int(minute_true))
	else:
		minute_true = str(int(minute_true))		
	hour = str(int(hour))
	# Format time
	time = str("{}:{}:{}.{}".format(hour,minute_true,second_true,decisecond_true))
	# Store time
	time_list.append(time)
	# Display time
	cv2.putText(img, time, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
	# Establish three sectors within the video frame.
	cv2.line(img, (640,0), (640,1040), (0,255,0), 1)
	cv2.line(img, (1280,0), (1280,1040), (0,255,0), 1)
	
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert imgage to grayscale
	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) # detect faces

	for (x,y,w,h) in faces:
		
		# Determine actor identity via face position 
		sector = 0
		bounding_box_midpoint = x+w/2
		if (bounding_box_midpoint < 640): # Target face belongs to actor 1 in sector 1
			sector = 1
		elif (bounding_box_midpoint < 1280): # Target face belongs to actor 2
			sector = 2
		else: # Target face belongs to actor 3
			sector = 3
		
		# Judge ID
		sector_list.append(sector)
		
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 3) #draw rectangle to main image
		
		# Mark the midpoint on each bounding box
		cv2.circle(img, ((int(bounding_box_midpoint)),(y+h)), 1, (0,0,225), 2)
		# Write sector numbers below the boxes
		cv2.putText(img, str(sector), (x, (y+h+15)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		# Create emotions array 
		info = np.array([frame_counter, sector])
		new_predictions = np.insert(predictions, 0, info)
		
		# Build lists for csv
		frame_list_F.append(frame_counter)
		angry_list.append(new_predictions[2])
		disgusted_list.append(new_predictions[3])
		afraid_list.append(new_predictions[4])
		happy_list.append(new_predictions[5])
		sad_list.append(new_predictions[6])
		surprised_list.append(new_predictions[7])
		neutral_list.append(new_predictions[8])
		# Coulmn 1 -> frame number, Column 2 -> sector,
		# Columns 3-9 -> emotion probabilities in previously listed order
      
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		
		emotion = emotions[max_index]
		
		#write emotion text above rectangle
		cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
		#process on detected face end		  			
		#-------------------------
	
	# Store frame
	out.write(img)
	
	# Display frame
	cv2.imshow('img',img)
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

J_FACS_df = pd.DataFrame(list(zip(sector_list,frame_list_F,angry_list,disgusted_list,afraid_list,happy_list,sad_list,surprised_list,neutral_list)),columns=['Sector','Frame','angry','disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral'])

# Ten rows per second, with timestamp
all_frames = pd.DataFrame(list(zip(frame_list,time_list,)),columns=['Frame','Time'])
J_FACS_df = pd.merge_ordered(all_frames,J_FACS_df,on='Frame',how='outer')

# Create and insert Seconds for later use in R
J_FACS_df.insert(1, 'Seconds', J_FACS_df['Frame']/30)

# Store data
J_FACS_df.to_csv('Judges_FACS.csv', index=False)


#kill open cv things		
cap.release()
out.release()
cv2.destroyAllWindows()
