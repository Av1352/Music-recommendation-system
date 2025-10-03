# Emotion-Based Music Recommendation System 🎼

A real-time music playlist recommender leveraging facial emotion recognition. Users are suggested playlists based on their detected mood via webcam, powered by deep learning.

🚩 Overview
Combining computer vision and music informatics, this project predicts user emotion from facial micro-expressions and recommends dynamic playlists tailored to their mood.

📊 Key Features
Real-time face capture and FER (Facial Emotion Recognition) using OpenCV

CNN-based emotion classifier trained on [name] dataset

Personalized music recommendation using Spotify API and custom playlists

Research-backed algorithm—includes published paper in repository

Flask web app interface (responsive UI)

🛠 Technologies Used
Python 3.9+

TensorFlow / Keras

OpenCV

pandas

Flask

Spotify Web API

HTML5/CSS3/JS (front-end)

📁 Project Structure
bash
music-recommendation-system/
├── app.py                # Web app controller
├── face_capture.py       # Webcam image capture
├── camera.py             # CNN-based emotion model
├── train.py              # ML training routines
├── templates/            # HTML for flask
├── songs/                # CSV music playlists by emotion
├── data/                 # Training images, metadata
├── requirements.txt      # Dependencies
├── research/
│   ├── paper.pdf         # Published work
│   ├── review_3.pdf      # Peer review
│   └── plagiarism_report.pdf
└── README.md
🚀 Getting Started
Clone the repo

bash
git clone https://github.com/Av1352/Music-recommendation-system.git
cd Music-recommendation-system
Install dependencies

bash
pip install -r requirements.txt
Run the app

bash
python app.py
# or use the Jupyter Notebook for exploration
🔥 Results
Emotion classifier accuracy: 96%

Music matching precision: 94%

Live demo: ![Screenshot or GIF](:** "Detected 'Happy' → Playlist: 'Feel Good Classics'"
