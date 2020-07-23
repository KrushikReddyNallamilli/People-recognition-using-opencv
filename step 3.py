import os
os.environ['PYTHONINSPECT'] = 'on'
import cv2
import numpy as np 
import pickle
#import RPi.GPIO as GPIO
from time import sleep

with open('labels', 'rb') as f:
	dicti = pickle.load(f)
	f.close()


camera = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(r"C:\Users\niceb\Downloads\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

font = cv2.FONT_HERSHEY_SIMPLEX
last=''

#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while True:
	ret,frame = camera.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	for (x, y, w, h) in faces:
		roiGray = gray[y:y+h, x:x+w]

		id_, conf = recognizer.predict(roiGray)

		for name, value in dicti.items():
			if value == id_:
				print(name)
				cv2.putText(frame, name, (x, y), font, 2, (0, 0 ,255), 2,cv2.LINE_AA)
				if name!=last :
					last=name
					
		if conf <= 70:
		
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow('frame', frame)
	key = cv2.waitKey(1)

	if key == 27:
		break

cv2.destroyAllWindows()
