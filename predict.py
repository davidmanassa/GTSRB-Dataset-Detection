import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image filenames")
args = vars(ap.parse_args())

filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

if "text/plain" == filetype:

	filenames = open(args["input"]).read().strip().split("\n")
	imagePaths = []

	for f in filenames:
		p = os.path.sep.join([config.IMAGES_PATH, f])
		imagePaths.append(p)

print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)

for imagePath in imagePaths:

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)

	preds = model.predict(image)[0]
	(startX, startY, endX, endY) = preds

	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

	cv2.imshow("Output", image)
	cv2.waitKey(0)