import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa.display
import pandas as pd

model_path = ("my_model.keras")
model = load_model(model_path)

feature_df = pd.DataFrame({"class": ["FAKE", "REAL"]})  

def extract_features(filename):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
    return mfccs_features_scaled.reshape(1, -1)

def detect_fake(filename):
    features_scaled = extract_features(filename)
    result_array = model.predict(features_scaled)

    print("Prediction Probabilities:", result_array)

    le = LabelEncoder().fit(feature_df["class"])
    print("Label Mapping:", le.classes_)

    if list(le.classes_) == ["FAKE", "REAL"]:
        result_classes = ["FAKE", "REAL"]
    else:
        result_classes = ["REAL", "FAKE"] 

    result = np.argmax(result_array[0])

    result_class = result_classes[result]

    print("Predicted Index:", result)
    print("Result:", result_class)

    return result_class, result_array[0]

st.title("🎙️ Audio Deepfake Detection")
st.write("Upload an audio file to detect if it is **REAL** or **FAKE**.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("⏳ Detecting...")
    result_class, result_array = detect_fake("temp_audio.wav")

    st.success(f"🎯 Prediction: **{result_class}**")

    sound_signal, sample_rate = librosa.load("temp_audio.wav", res_type="kaiser_fast")
    plt.figure(figsize=(12, 4))
    plt.title("Audio Waveform")
    plt.plot(sound_signal)
    st.pyplot(plt)

    plt.figure(figsize=(14, 5))
    spec = np.abs(librosa.stft(sound_signal))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec_db, sr=sample_rate, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    st.pyplot(plt)







# python -m streamlit run app.py