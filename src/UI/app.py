import streamlit as st
import numpy as np
import cv2
import mlflow.keras
import os
import sys

# Add modeling folder to the Python path
current_dir = os.path.dirname(__file__)
modeling_path = os.path.abspath(os.path.join(current_dir, "..", "modeling"))
if modeling_path not in sys.path:
    sys.path.insert(0, modeling_path)

from preprocessing import preprocess_image  # noqa: E402
from Crack_intensity import process_crack_intensity  # noqa: E402

# Set experiment and model URI
EXPERIMENT_NAME = "Surface Crack Detection Experiments"
MODEL_NAME = "Preprocess Custom CNN"
MODEL_VERSION = 2
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


# Load the model once at the start
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(
        "https://dagshub.com/Saipr14/ML-Surface-Crack-Detection.mlflow"
    )
    return mlflow.keras.load_model(MODEL_URI)


model = load_model()

# Streamlit app
st.title("🧱 Surface Crack Detection")
st.write("Upload an image to detect whether it contains a surface crack.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Ensure uploaded_file is valid before proceeding
    if img is None:
        st.error("Error loading image. Please try a different image.")
    else:
        # Convert to RGB if grayscale or single-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.merge([img, img, img])

        # Preprocess
        preprocessed = preprocess_image(img)
        resized = cv2.resize(preprocessed, (96, 96))
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        # Prediction
        prediction = model.predict(input_tensor)
        prob = prediction[0][0]
        predicted_class = 0 if prob > 0.5 else 1
        class_mapping = {0: "Non-Crack", 1: "Crack"}

        # Show intensity image if crack detected
        if predicted_class == 1:
            intensity_image = process_crack_intensity(resized)
            st.image(
                intensity_image, caption="Crack Intensity Visualization", channels="BGR"
            )

        # Display processed image and prediction
        st.image(resized, caption="Processed Image")
        st.markdown(f"### 🔍 Prediction: **{class_mapping[predicted_class]}**")
        st.markdown(f"**Confidence:** `{prob:.4f}`")
