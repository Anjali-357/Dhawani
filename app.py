import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import io

# Set page config
st.set_page_config(
    page_title="Audio Deepfake Detector",
    layout="centered",
    page_icon="🔊"
)

# App title
st.title("🔍 DHWANI Audio Deepfake Detector")
st.markdown("When AI Listen, DHWANI Detects!")
st.markdown("Upload an audio file and let the AI detect whether it's *Real* or a *Deepfake* 🕵️‍♂️")

# Function to extract MFCC features
def extract_features_from_audio(audio_bytes, max_length=500, sr=16000, n_mfcc=40):
    try:
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)

        # Pad or trim
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]

        # Reshape
        return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('updated_model.h5')

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Upload file
uploaded_file = st.file_uploader("🎵 Upload Audio File", type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Show Detect Button
    if st.button("🧠 Detect Deepfake"):
        if model is not None:
            features = extract_features_from_audio(uploaded_file.getvalue())
            if features is not None:
                with st.spinner('Analyzing audio...'):
                    prediction = model.predict(features)
                    confidence = prediction[0][0]

                st.subheader("📊 Detection Results")

                # Confidence and label
                is_real = confidence > 0.5
                confidence_percentage = confidence * 100 if is_real else (1 - confidence) * 100

                # Results display
                col1, col2 = st.columns(2)
                if is_real:
                    col1.success("✅ *Real Audio*")
                    col2.metric("Confidence", f"{confidence_percentage:.2f}%", delta="↑")
                else:
                    col1.error("🚨 *Deepfake Audio*")
                    col2.metric("Confidence", f"{confidence_percentage:.2f}%", delta="↓")

                # Confidence bar
                st.markdown("### 📈 Confidence Level")
                st.progress(float(confidence if is_real else 1 - confidence))
        else:
            st.error("Model could not be loaded. Please check your model path.")

# Sidebar
with st.sidebar:
    st.header("📘 About the App")
    st.markdown(
        """
        This app uses a *CNN-BiLSTM* deep learning model to detect whether an uploaded audio is *real* or *deepfake*.
        
        The model processes the audio's *MFCC features* and classifies based on learned speech patterns.
        """
    )
    
    st.header("⚙️ How to Use")
    st.markdown(
        """
        1. Upload an audio file (.wav, .mp3, or .ogg)  
        2. Click on *Detect Deepfake*  
        3. View the detection result and confidence level
        """
    )

    st.caption("Developed with ❤️ using Streamlit")