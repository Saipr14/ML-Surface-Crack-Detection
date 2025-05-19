import streamlit as st
import numpy as np
import cv2
from src.modeling.Crack_intensity import predict_image
# Later: from src.modeling.intensity import calculate_intensity

st.title("🧱 Crack Detection & Intensity Analysis")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    try:
        prediction, confidence = predict_image(img)
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")
        # Later: st.warning(f"Crack Intensity: {calculate_intensity(...)}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Please ensure the image is in the correct format and try again.")
