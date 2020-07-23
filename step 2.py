import os
import numpy as np 
from PIL import Image 
import cv2
import pickle
#import serial

#ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

faceCascade = cv2.CascadeClassifier(r"C:\Users\niceb\Downloads\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml")
#this harcascade file is uploaded separetely
recognizer = cv2.face.LBPHFaceRecognizer_create()

baseDir = os.path.dirname(os.path.abspath(__file__))

#train the images under the images folder
imageDir = os.path.join(baseDir, "D:\programs\opencv\people recognization\images")

currentId = 1
labelIds = {}
yLabels = []
xTrain = []
#ser.write("Training.....".encode())

for root, dirs, files in os.walk(imageDir):
	print(root, dirs, files)
	for file in files:
		print(file)
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root)
			print(label)

			if not label in labelIds:
				labelIds[label] = currentId
				print(labelIds)
				currentId += 1

			id_ = labelIds[label]
			pilImage = Image.open(path).convert("L")
			imageArray = np.array(pilImage, "uint8")
			faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)

			for (x, y, w, h) in faces:
				roi = imageArray[y:y+h, x:x+w]
				xTrain.append(roi)
				yLabels.append(id_)

with open("labels", "wb") as f:
	pickle.dump(labelIds, f)
	f.close()

recognizer.train(xTrain, np.array(yLabels))
recognizer.save("trainer.yml")
print(labelIds)
