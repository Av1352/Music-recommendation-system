import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# ---------- App Theme + Style ----------
# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f3f6fb;
        }
        .main {
            background-color: #ffffff;
            border-radius: 18px;
            padding: 2rem;
            box-shadow: 0 6px 24px 0 rgba(34, 42, 64, 0.09);
        }
        .stButton>button {
            background-color: #3b5ee3;
            color: white;
            border: None;
            border-radius: 5px;
            padding: 8px 24px;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #233187;
            color: #fff;
        }
        .emotion-card {
            background: linear-gradient(90deg, #eaf6fb 0%, #f6e8fc 100%);
            padding: 24px 16px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 16px 0 rgba(60,98,132,.07);
        }
        .footer {
            text-align: center; color: #8c98ad; font-size: 16px; margin-top: 32px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------- Settings & Logo -------------
logo_path = "streamlit_app/Logo.png"
st.set_page_config(page_title="Music Recommender by Emotion", page_icon=logo_path, layout="centered")
st.image(logo_path, width=120)
st.markdown("<h2 style='text-align:center; color:#455d7a;'>Emotion-Based Music Recommendation</h2>", unsafe_allow_html=True)

# ------------- Sidebar -------------
with st.sidebar:
    st.image(logo_path, width=80)
    st.markdown("Upload a face photo *or* use the webcam, and get a custom playlist for your emotion!\n\n[GitHub Repo](https://github.com/Av1352/Music-recommendation-system)")
    st.write("---")
    st.markdown("**Project by Av1352**")
    st.write("")

# ---------- Model Architecture -----------
def build_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('model.h5')
    return model

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_image(image):
    img = image.convert('L')
    img = img.resize((48, 48))
    img = np.array(img) / 255.0
    img = np.reshape(img, (1, 48, 48, 1))
    return img

def predict_emotion(image, model):
    img = preprocess_image(image)
    preds = model.predict(img)
    idx = np.argmax(preds)
    return EMOTION_LABELS[idx], float(np.max(preds)) * 100

# ---------- Main App UI -----------
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.write("#### Upload your face photo or capture from webcam")
    uploaded_img = st.file_uploader("Upload a .jpg/.png face image", type=["jpg", "png", "jpeg"])
    webcam_img = None

    if st.button("📸 Use Webcam (Streamlit Cloud only)"):
        webcam_img = st.camera_input("Take a picture")

    if uploaded_img or webcam_img:
        if uploaded_img:
            image = Image.open(uploaded_img)
        else:
            image = Image.open(webcam_img)
        st.image(image, caption="Your face photo", width=210)

        model = build_emotion_model()
        with st.spinner("🎯 Analyzing your emotion..."):
            emotion, confidence = predict_emotion(image, model)

        st.markdown(
            f"<div class='emotion-card'><h4 style='margin-top:0;color:#41c2b7;'>Emotion detected: {emotion.capitalize()} &nbsp; <span style='font-size:17px;'>({confidence:.1f}% confidence)</span></h4></div>",
            unsafe_allow_html=True
        )

        playlist_path = f"songs/{emotion.lower()}.csv"
        if os.path.exists(playlist_path):
            playlist = pd.read_csv(playlist_path)
            st.markdown(f"##### 🎵 Playlist for {emotion.capitalize()}")
            st.dataframe(playlist[["Name", "Album", "Artist"]].head(15), use_container_width=True)
        else:
            st.warning("No playlist found for this emotion.")

    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("ℹ️ How it works & Tech stack", expanded=False):
    st.markdown("""
    1. Upload or capture a face photo.
    2. Your emotion is recognized using a deep CNN trained on facial expressions.
    3. A custom music playlist is curated for your mood!
    ---
    **Built with:** TensorFlow/Keras | OpenCV | Streamlit | pandas
    """)

st.markdown(
    "<div class='footer'>Made with ❤️ by Av1352 &nbsp;|&nbsp; <a href='https://github.com/Av1352/Music-recommendation-system' style='color:#3b5ee3;' target='_blank'>GitHub Repo</a></div>",
    unsafe_allow_html=True
)
