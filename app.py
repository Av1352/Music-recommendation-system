from flask import Flask
from flask import render_template, request, url_for, redirect, session, make_response, flash
import sqlite3 as lite
import os
import cv2

from deepface import DeepFace
from fer import FER
import face_capture as face

app = Flask(__name__)
app_root = os.path.abspath(os.path.dirname(__file__))

app.secret_key = os.urandom(10)

@app.route('/')
def index():

	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)