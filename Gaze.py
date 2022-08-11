# Code for expression extraction taken from github.com/serengil/tensorflow-101
# Code for gaze and blink tracking taken from pysource.com/2019/01/17/eye-gaze-detection-2-gaze-controlled-keyboard-with-python-and-opencv-p-4/



# Run CSVs through add_Session_to_csv.py to complete formatting
###############################################################

# Import necessary libraries
#import tensorflow
import numpy as np
import cv2
from keras.preprocessing import image
from os import listdir
import dlib
from math import hypot
import pandas as pd

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')

#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("lib/tensorflow-101-master/model/facial_expression_model_structure.json", "r").read())
model.load_weights('lib/tensorflow-101-master/model/facial_expression_model_weights.h5') #load weights
#-----------------------------
#-----------------------------

emotions = ('Angry', 'Disgusted', 'Afraid', 'Happy', 'Sad', 'Surprised', 'Neutral')

#---------------------------------------------------------------------------#
# Facial Landmarks for Gaze/Blink reading
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lib/shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
	return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

#------------------------------
def get_blinking_ratio(eye_points, facial_landmarks):
	left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
	right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
	center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
	center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
	hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
	ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
	
	if ver_line_lenght <= 0:
		return hor_line_lenght
	
	ratio = hor_line_lenght / ver_line_lenght	
	return ratio

#------------------------------
def get_gaze_ratio(eye_points, facial_landmarks):
	left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
										(facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
										(facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
										(facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
										(facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
										(facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

	height, width, _ = img.shape
	mask = np.zeros((height, width), np.uint8)
	cv2.polylines(mask, [left_eye_region], True, 255, 2)
	cv2.fillPoly(mask, [left_eye_region], 255)
	eye = cv2.bitwise_and(gray, gray, mask=mask)

	min_x = np.min(left_eye_region[:, 0])
	max_x = np.max(left_eye_region[:, 0])
	min_y = np.min(left_eye_region[:, 1])
	max_y = np.max(left_eye_region[:, 1])

	gray_eye = eye[min_y: max_y, min_x: max_x]
	_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
	height, width = threshold_eye.shape
	left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
	left_side_white = cv2.countNonZero(left_side_threshold)

	right_side_threshold = threshold_eye[0: height, int(width / 2): width]
	right_side_white = cv2.countNonZero(right_side_threshold)

	if left_side_white == 0:
		gaze_ratio = 0
	elif right_side_white == 0:
		gaze_ratio = left_side_white
	else:
		gaze_ratio = left_side_white / right_side_white
	
	return gaze_ratio
#---------------------------------------------------------------------------#

# Get all mp4 files from directory
def find_mp4_filenames( path_to_dir, suffix=".mp4" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

# Path to Participants_FACS.py. Facial videos should be stored in the same folder. No mp4 files other than facial videos should be stored in this folder.
filenames = find_mp4_filenames("Input/")

for name in filenames:
	p_id = name[7:11]
	#sess = name.split("_")[2]
	
	framelocation = 'Frames' + '/' + p_id

	if not os.path.exists(framelocation):
		os.makedirs(framelocation)
	
	newmp4 = 'Output/' + p_id + '_Gaze.mp4'
	newcsv = 'CSV/Subjects/' + p_id + '_Gaze.csv'
	# Input video file
	cap = cv2.VideoCapture("Input/" + name)
	# Output mp4
	out = cv2.VideoWriter(newmp4, 25, 10.0, (int(cap.get(3)), int(cap.get(4))))
	#-----------------------------
	
	# Pseudo timekeeper
	frame_counter = 0
	frame_list = []
	# Prepare to build FACS+Gaze df
	frame_list_F = []
	frame_list_G = []
	#------------------
	angry_list = []
	disgusted_list = []
	afraid_list = []
	happy_list = []
	sad_list = []
	surprised_list = []
	neutral_list = []
	#------------------
	blink_list = [] # represented as WOR in the csv
	ratio_list = []
	direction_list = []
	#------------------
	# Coordinates and Diminsions for both bounding boxes per face at each frame
	F_x1_list = []
	F_x2_list = []
	F_y1_list = []
	F_y2_list = []
	
	F_height_list = []
	F_width_list = []
	F_area_list = []
	
	G_x1_list = []
	G_x2_list = []
	G_y1_list = []
	G_y2_list = []
	
	G_height_list = []
	G_width_list = []
	G_area_list = []
	
	while(True):
		ret, img = cap.read() # read frame from file
		
		if np.shape(img) == (): # Stop reading at end of file
			break
		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert imgage to grayscale
		
		faces_F = face_cascade.detectMultiScale(gray, 1.3, 5) # detect faces
				
		# Update frame number
		frame_counter += 1
		frame_list.append(frame_counter)								
		# Update and display seconds
		seconds = frame_counter / 10
		cv2.putText(img, str(seconds)+" s", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		# Display Participant id
		cv2.putText(img, p_id, (557, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
		# FACS
		for (x,y,w,h) in faces_F:
			
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			
			# Store coordinates per face at frame
			F_x1_list.append(x)
			F_x2_list.append(x+w)
			F_y1_list.append(y)
			F_y2_list.append(y+h)
			# Store dimensions per face at frame
			F_height_list.append(h)
			F_width_list.append(w)
			F_area_list.append(h*w)
	
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			
			# Create emotions array
			info = np.array([int(frame_counter)]) # One row per frame
			new_predictions = np.insert(predictions, 0, info)
			
			# Build lists for the .csv file. These lists become columns
			# List 1 -> frame number
			# Lists 2-8 -> emotion probabilities in previously listed order
			frame_list_F.append(frame_counter)
			angry_list.append(new_predictions[1])
			disgusted_list.append(new_predictions[2])
			afraid_list.append(new_predictions[3])
			happy_list.append(new_predictions[4])
			sad_list.append(new_predictions[5])
			surprised_list.append(new_predictions[6])
			neutral_list.append(new_predictions[7])
			
		#---------------------------------------------------------------------#
		# Gaze
		faces_G = detector(gray)

		try: # Prevents crasehs when a face is partially out of frame	
			for face in faces_G:
				G_x1, G_y1 = face.left(), face.top()
				G_x2, G_y2 = face.right(), face.bottom()
				#cv2.rectangle(img, (G_x1, G_y1), (G_x2, G_y2), (0, 255, 0), 2)
				
				# Store coordinates per face at frame
				G_x1_list.append(G_x1)
				G_x2_list.append(G_x2)
				G_y1_list.append(G_y1)
				G_y2_list.append(G_y2)
				# Calculate dimension
				height = G_y2 - G_y1
				width = G_x2 - G_x1
				area = height * width
				# Sore dimensions per face at frame
				G_height_list.append(height)
				G_width_list.append(width)
				G_area_list.append(area)
		
				landmarks = predictor(gray, face) # 68 facial landmarks. This is a common system; information is easily available online
				
				left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
				right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
				blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
				
				# Store openness ratio for all frames
				frame_list_G.append(frame_counter)
				blink_list.append(blinking_ratio)
				
				if blinking_ratio > 5.7: # Don't check gaze ratio if eyes are closed
					direction = "CLOSED"
					direction_list.append(direction)
					ratio_list.append("")
					cv2.putText(img, direction, (5, 475), font, 2, (0, 0, 255), 2)
					break
		
				# Gaze detection
				gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
				gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
				gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
		
				if gaze_ratio <= 1:
					direction = "RIGHT"
				elif 1 < gaze_ratio < 1.7:
					direction = "CENTER"
				else:
					direction = "LEFT"
					
				# Build lists			
				direction_list.append(direction)
				ratio_list.append(gaze_ratio)
				
				# Show gaze
				#cv2.putText(img, str(gaze_ratio), (5, 470), font, 2, (0, 0, 255), 2)
				# Show direction
				cv2.putText(img, direction, (5, 475), font, 2, (0, 255, 0), 2)
				
		except AttributeError:
			pass
		#-----------------
		fname = framelocation + '/' + str(frame_counter) + '.jpg'

		# Store frame
		cv2.imwrite(fname, img)
		
		# Display frame
		# cv2.imshow('img',img)
		# if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		# 	break
		
	#FACS_df = pd.DataFrame(list(zip(frame_list_F,angry_list,disgusted_list,afraid_list,happy_list,sad_list,surprised_list,neutral_list,F_x1_list,F_x2_list,F_y1_list,F_y2_list,F_height_list,F_width_list,F_area_list)),columns=['Frame','F_Angry','F_Disgusted','F_Afraid','F_Happy','F_Sad','F_Surprised','F_Neutral','F_x1','F_x2','F_y1','F_y2','F_height','F_width','F_area'])
	Gaze_df = pd.DataFrame(list(zip(frame_list_G,ratio_list,direction_list,blink_list,G_x1_list,G_x2_list,G_y1_list,G_y2_list,G_height_list,G_width_list,G_area_list)),columns=['Frame','G_Ratio','G_Direction','G_WOR','G_x1','G_x2','G_y1','G_y2','G_height','G_width','G_area'])
	
	# G_height_list,G_width_list,G_area_list
	
	# Combine FACS and Gaze
	#FG_df = pd.merge_ordered(FACS_df, Gaze_df,on='Frame',how='outer')
	
	# Ten rows per second
	all_frames = pd.DataFrame(frame_list,columns=['Frame'])
	FG_df = pd.merge_ordered(Gaze_df,all_frames,on='Frame',how='outer')
	
	# Create and insert Seconds for later use in R
	FG_df.insert(1, 'Seconds', FG_df['Frame']/10)
	# Give participants their names
	FG_df.insert(0, 'Participant_ID', p_id)
	
	
	# Store data
	FG_df.to_csv(newcsv, index=False)
	
#kill open cv things		
cap.release()
out.release()
cv2.destroyAllWindows()