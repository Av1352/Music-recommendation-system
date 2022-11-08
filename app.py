from flask import Flask
from flask import render_template, request, url_for, redirect, session, make_response, flash
import os
import cv2
import pandas as pd
from fer import FER
import face_capture as face
import camera as camera
import Spotipy as s
app = Flask(__name__)
app_root = os.path.abspath(os.path.dirname(__file__))

app.secret_key = os.urandom(10)

headings = ("Name","Album","Artist")
df1 = camera.music_rec()
df1 = df1.head(15)

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')

@app.route('/click')
def click():
	face.capture()
	img = cv2.imread("user.png")
	
	# predictions = DeepFace.analyze(img)
	# print(predictions)
	prediction = camera.camera()
	print(prediction)
	
	global emotion
	emotion = prediction
	
	return render_template('index.html', item=prediction)
	
@app.route('/generate')
def gen_table():
    return df1.to_json(orient='records')



if __name__ == '__main__':
    app.run(debug=True)