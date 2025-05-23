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

# ===================== UI Layout =====================

st.markdown(
    "<h2 style='text-align: center; color: steelblue;'>🧠 Smart Surface Crack Detector</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center;'>Upload a surface image to analyze and classify cracks based on intensity.</h4>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Model Info
with st.expander("ℹ️ Model Information"):
    st.markdown("""
    - Model: Custom CNN (Logged via MLflow)
    - Preprocessing: Grayscale → Gaussian Blur → Adaptive Edge Detection
    - Crack Intensity Classification:
        - **0** - Low  
        - **1** - Medium  
        - **2** - High
    """)

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Error loading image. Please try a different one.")
    else:
        # Ensure image is 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.merge([img, img, img])

        # Preprocessing
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
            i_image = cv2.resize(resized, (227, 227))
            intensity_image = process_crack_intensity(i_image)

            st.markdown("---")
            st.subheader("🎯 Crack Intensity Visualization")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    intensity_image,
                    caption="🛠️ Intensity Map",
                    channels="BGR",
                    use_container_width=True,
                )
        else:
            st.markdown("---")
            st.markdown(
                "<h5 style='text-align: center; color: gray;'>No cracks detected in the image.</h5>",
                unsafe_allow_html=True,
            )
            st.subheader("📸 Preprocessed Input Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    resized,
                    caption="🛠️ Processed Image",
                    channels="gray",
                    use_container_width=True,
                )

        # Display result
        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        if predicted_class == 1:
            st.success(f"✅ Prediction: **{class_mapping[predicted_class]}**")
        else:
            st.warning(f"🧱 Prediction: **{class_mapping[predicted_class]}**")

        st.info(f"📊 Confidence Score: `{prob:.4f}`")

        st.markdown("---")
        st.markdown(
            "<h5 style='text-align: center; color: gray;'>© 2025 Automated Surface Crack Detection</h5>",
            unsafe_allow_html=True,
        )
