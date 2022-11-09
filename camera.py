import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
import Spotipy
import time
import pandas as pd
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fear",
#                     3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {0: "songs/angry.csv", 1: "songs/disgusted.csv ", 2: "songs/fearful.csv",
                  3: "songs/happy.csv", 4: "songs/neutral.csv", 5: "songs/sad.csv", 6: "songs/surprised.csv"}
show_text = [0]
def camera():
	face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	ds_factor = 0.6
	emotion_model = Sequential()
	emotion_model.add(Conv2D(32, kernel_size=(
        3, 3), activation='relu', input_shape=(48, 48, 1)))
	emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
	emotion_model.add(Dropout(0.25))
	emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
	emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
	emotion_model.add(Dropout(0.25))
	emotion_model.add(Flatten())
	emotion_model.add(Dense(1024, activation='relu'))
	emotion_model.add(Dropout(0.5))
	emotion_model.add(Dense(7, activation='softmax'))
	emotion_model.load_weights('model.h5')

	cv2.ocl.setUseOpenCL(False)
	emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fear",
                    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
	# music_dist = {0: "songs/angry.csv", 1: "songs/disgusted.csv ", 2: "songs/fearful.csv",
    #               3: "songs/happy.csv", 4: "songs/neutral.csv", 5: "songs/sad.csv", 6: "songs/surprised.csv"}
	# show_text = [0]
	image = cv2.imread("user.png")
	image = cv2.resize(image, (600, 500))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

	for  (x, y, w, h) in face_rects:
		cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
		roi_gray_frame = gray[y:y + h, x:x + w]
		cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
		prediction = emotion_model.predict(cropped_img)
		global maxindex
		maxindex = int(np.argmax(prediction))
		show_text[0]=maxindex
		print(show_text)
		print(emotion_dict[maxindex])
		global emo
		emo = emotion_dict[maxindex]
		return emo

def music_rec():
	# print('---------------- Value ------------', music_dist[show_text[0]])
	
	df = pd.read_csv(music_dist[show_text[0]])
	df = df[['Index','Name','Album','Artist']]
	df = df.head(15)
	return df

if __name__ == '__main__':
    camera()