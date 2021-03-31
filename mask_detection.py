from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import os
import cv2

# Setting up the detect and facial landmark modules
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Read from camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

flag = 0

while True:

	# Read from camera
	frame = vs.read()

	# Resize to increase damage due to distress
	frame = imutils.resize(frame, width=700)

	# convert frame to greyscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces
	faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,		minNeighbors=5, minSize=(100, 100),		flags=cv2.CASCADE_SCALE_IMAGE)

	# iterate faces
	for (x, y, w, h) in faces:

		# look around the face
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

		# detect of landmarks
		landmark = landmark_detect(gray, rect)
		landmark = face_utils.shape_to_np(landmark)

		#  mouth landmark
		(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		mouth = landmark[mStart:mEnd]

		# draw rectangle on frame
		boundRect = cv2.boundingRect(mouth)
		cv2.rectangle(frame,
					  (int(boundRect[0]), int(boundRect[1])),
					  (int(boundRect[0] + boundRect[2]),  int(boundRect[1] + boundRect[3])), (0,0,255), 2)
        

		# Average saturation calculation
		hsv = cv2.cvtColor(frame[int(boundRect[1]):int(boundRect[1] + boundRect[3]),int(boundRect[0]):int(boundRect[0] + boundRect[2])], cv2.COLOR_RGB2HSV)
		sum_saturation = np.sum(hsv[:, :, 1])
		area = int(boundRect[2])*int(boundRect[3])
		avg_saturation = sum_saturation / area
        

		# Check and close with threshold
		if avg_saturation>100:
			cv2.putText(frame, "Not Wearing Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
						2)

	# Show frame on screen
	cv2.imshow("Camera", frame)

	# press Esc to quit
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break


cv2.destroyAllWindows()
vs.stop()