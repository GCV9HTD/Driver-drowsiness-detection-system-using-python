import cv2
import numpy as np 

# Create our body classifier
body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Open the video
cap = cv2.VideoCapture(0)  #"videoplayback.mp4"

# loop once video is successfully loaded
while True:
	# Read first frame
	ret, frame = cap.read()

	# This is necessary so that we don't get errors when the video is completed
	if ret == False:
		break

	frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	bodies = body_classifier.detectMultiScale(gray,1.1,1)

	# Extract bounding boxes for any bodies identified
	for (x,y,w,h) in bodies:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		cv2.imshow("Pedestrians",frame)

	if cv2.waitKey(1) == 13:
		break

cap.release()
cv2.destroyAllWindows()
