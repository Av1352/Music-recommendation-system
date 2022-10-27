from flask import Flask
from flask import render_template, request, url_for, redirect, session, make_response, flash
import os
import cv2

from deepface import DeepFace
from fer import FER
import face_capture as face

app = Flask(__name__)
app_root = os.path.abspath(os.path.dirname(__file__))

app.secret_key = os.urandom(10)

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')

@app.route('/click')
def click():
	face.capture()
	img = cv2.imread("user.png")
	predictions = DeepFace.analyze(img)
	print(predictions)
	
	global emotion
	emotion = predictions['dominant_emotion']
	
	return render_template('index.html', predictions=predictions['dominant_emotion'])
	
@app.route('/generate')
def generate():
	return render_template('generate.html', predictions = emotion)

if __name__ == '__main__':
    app.run(debug=True)