# USAGE
# python detect_yawning.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
import cv2
import imutils
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse

import time
import dlib

import pygame


def sound_alarm(path):
    # play an alarm sound
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    #A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    #C= dist.euclidean(mouth[3], mouth[5])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0], mouth[4])

    # compute the eye aspect ratio
    #mar = (A+B+C) / (3.0 * D)
    mar = (B) / ( D)
    #ear=(A) / (C)
    # return the eye aspect ratio
    return mar


# construct the argument parse and parse the arguments


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path alarm .WAV file")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.8
EYE_AR_CONSEC_FRAMES = 30

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
#(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        #leftEye = shape[lStart:lEnd]
        #rightEye = shape[rStart:rEnd]
        mouth=shape[mStart:mEnd]
        #leftEAR = eye_aspect_ratio(leftEye)
        #rightEAR = eye_aspect_ratio(rightEye)
        mouthAR =mouth_aspect_ratio(mouth)


        # average the eye aspect ratio together for both eyes
        #ear = (leftEAR + rightEAR) / 2.0
        mar = mouthAR
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        #leftEyeHull = cv2.convexHull(leftEye)
        #rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if mar > EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
