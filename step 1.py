import cv2

import numpy as np 
import os
import sys
import time

camera = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(r'C:\Users\niceb\Downloads\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml') #add your harcascade file path

name = input("What's his/her Name? ")

#all the files will be saved under the specified folder

dirName = "D:\programs\opencv\people recognization\images\ " + name
print(dirName)

if not os.path.exists(dirName):
	os.makedirs(dirName)
	print("Directory Created")
	
else:
	print("Name already exists")
	
	sys.exit()

count = 1

#we are going to collect 30 samples
while count < 31:
# for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	ret,frame = camera.read()
	if count > 30:
		break
	# frame = frame.array
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.5, 5)
	for (x, y, w, h) in faces:
		roiGray = gray[y:y+h, x:x+w]
		fileName = dirName + "/" + name + str(count) + ".jpg"
		cv2.imwrite(fileName, roiGray)
		cv2.imshow("face", roiGray)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		time.sleep(1)
		count += 1
	cv2.imshow('frame', frame)
	key = cv2.waitKey(1)
	

	if key == 27:
		break

	#camera.release()
	cv2.destroyAllWindows()

#A
