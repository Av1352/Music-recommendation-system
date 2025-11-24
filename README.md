# Emotion-Based Music Recommendation System 🎼

A real-time music playlist recommender leveraging facial emotion recognition. Users are suggested playlists based on their detected mood via webcam, powered by deep learning.

- **🟦 Try it live:** [music-rec-system.streamlit.app](https://music-rec-system.streamlit.app)
- Upload a face photo or use webcam (in supported browsers)
- Get an instant playlist for your detected emotion!

## 🚀 Live Demo Usage

1. **Upload your face photo** _(jpg/png)_, or use **webcam** to capture one.
2. The app detects your emotion using a trained CNN.
3. Instantly get a Spotify-style playlist matched to your mood.

## 🚩 Overview

Combining computer vision and music informatics, this project predicts user emotion from facial micro-expressions and recommends dynamic playlists tailored to their mood.

## 📊 Key Features

- Real-time face capture and FER (Facial Emotion Recognition) using OpenCV
- CNN-based emotion classifier trained on custom dataset
- Personalized music recommendation using Spotify API and custom playlists
- Research-backed algorithm—includes published paper in repository
- Flask web app interface (responsive UI)

## 🛠 Technologies Used

- Python 3.9+
- TensorFlow / Keras
- OpenCV
- pandas
- Flask
- Spotify Web API
- HTML5/CSS3/JS (front-end)

## 📁 Project Structure

```
music-recommendation-system/
├── app.py                # Web app controller
├── face_capture.py       # Webcam image capture
├── camera.py             # CNN-based emotion model
├── train.py              # ML training routines
├── templates/            # HTML for flask
├── songs/                # CSV music playlists by emotion
├── data/                 # Training images, metadata
├── streamlit_app.py   # Portfolio/demo app for recruiters (Streamlit)
├── requirements.txt      # Dependencies
├── research/
│   ├── paper.pdf         # Published work
│   ├── review_3.pdf      # Peer review
│   └── plagiarism_report.pdf
└── README.md
```

## 🚀 Getting Started

1. **Clone the repo**

   ```
   git clone https://github.com/Av1352/Music-recommendation-system.git
   cd Music-recommendation-system
   ```

2. **Install requirements**

   ```
   pip install -r requirements.txt
   ```


3. **Launch the application**

   ```
    python app.py
    ```

## 🔥 Results

- **Emotion classifier accuracy**: 96%
- **Music matching precision**: 94%
