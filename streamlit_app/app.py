import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# ------ CONFIG ------
logo_path = "streamlit_app/Logo.png"  # Place your logo in the same folder as this script
st.set_page_config(page_title="Music Recommender by Emotion", page_icon=logo_path, layout="centered")

st.image(logo_path, width=120)
st.markdown("## 🎧 Emotion-Based Music Recommendation", unsafe_allow_html=True)

# ------ SIDEBAR ------
with st.sidebar:
    st.image(logo_path, width=80)
    st.title("Music Recommender 🎼")
    st.info("Upload a face photo **or** use webcam.\n\nGet a custom playlist for your emotion!\n\n[GitHub Repo](https://github.com/Av1352/Music-recommendation-system)")
    st.write("---")
    st.caption("Project by Av1352")

# ------ MODEL DEFINITION ------
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

# ------ MAIN UI ------
st.write("#### Upload your face photo or use webcam")
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

    st.success(f"**Emotion detected:** {emotion.capitalize()} ({confidence:.1f}% confidence)")

    playlist_path = f"songs/{emotion.lower()}.csv"
    if os.path.exists(playlist_path):
        playlist = pd.read_csv(playlist_path)
        st.write(f"##### 🎵 Playlist for {emotion.capitalize()}")
        st.dataframe(playlist[["Name", "Album", "Artist"]].head(15), use_container_width=True)
    else:
        st.warning("No playlist found for this emotion.")

with st.expander("How it works & Tech stack", expanded=False):
    st.write("""
        1. Upload or capture a face photo.
        2. Your emotion is recognized with a deep CNN.
        3. You get a custom music playlist for your mood!
        ---
        **Built with:** TensorFlow/Keras | OpenCV | Streamlit | pandas
    """)

st.caption("Made with ❤️ by Av1352 | [GitHub Repo](https://github.com/Av1352/Music-recommendation-system)")
