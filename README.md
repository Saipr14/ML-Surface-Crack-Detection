# Final year Machine Learning Project
```bash
# Note: I have only forked the file structure from datalumina and I also didn't exactly used the whole structure.I used the ones that I needed,
So don't think this project is copied fully from that source."It is not".
```
# 🧱 Surface Crack Detection and Intensity Classification

This project implements an automated surface crack detection system using a custom-trained Convolutional Neural Network (CNN) along with a rule-based crack intensity classifier. The system analyzes grayscale concrete images to detect cracks and assess their severity, providing actionable insights for structural maintenance teams.

## 📂 Dataset

We used the **[Surface Crack Detection Dataset from Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)**. The dataset contains over 40,000 labeled grayscale images of concrete surfaces divided into:
- `Positive` (images with cracks)
- `Negative` (images without cracks)

The images are typically sized 227x227 pixels and are suitable for deep learning-based detection tasks.

## 🧠 Features & Techniques

- 🔍 **CNN-Based Detection**: A custom Convolutional Neural Network trained to classify images into 'Crack' and 'Non-Crack'.
- 🛠️ **Image Preprocessing**: Includes Gaussian Blur,Bilateral Filter, Adaptive Canny edge detection,Custom Percolation fill and Morphological operations.
- 📏 **Crack Intensity Classification**:
  - Crack area vs image area (crack ratio)
  - Local width of the crack (from bounding box)
  - Intensity scored as:
    - `0 – Low`
    - `1 – Medium`
    - `2 – High`
- 🌈 **Visual Feedback**: Crack intensity overlaid on images using bounding boxes colored:
  - Green = Low
  - Yellow = Medium
  - Red = High

## 📦 Project Structure
```bash
├── app
│ ├── main_app.py # Streamlit frontend
├── modeling
│ ├── preprocess_image.py # Preprocessing pipeline
│ ├── Crack_intensity.py # Custom crack intensity logic
├── models
│ └── (MLflow hosted model)
├── README.md
└── requirements.txt
```
## 🚀 Launching the Streamlit App

### 🔧 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/surface-crack-detection.git
cd surface-crack-detection
```
### 📦 Step 2: Install Dependencies
It’s recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### 🎯 Step 3: Launch the Streamlit App
```bash
streamlit run app/main_app.py
```
Then open the displayed localhost URL in your browser (usually http://localhost:8501).

## 🖼️ Testing with Images
Upload any concrete surface image (preferably 227x227 grayscale or RGB).

If a crack is detected, it will be highlighted with a colored intensity label.

Final prediction and intensity classification will be displayed on the UI.

## 🔗 Model Tracking & Logging
We use MLflow to version and load our CNN model.

The model is hosted remotely via DagsHub.

## 📌 Notes
No login or authentication is required to use the app.

Entire logic is built on rule-based analysis for intensity rather than using separate classification datasets.

## ✨ Future Improvements
Deploy as a web API for integration with drone/robot inspection systems.

Extend to real-time video stream analysis.

Support larger image resolutions for fine-grained inspection.

## 👨‍💻 Authors
Saipr14 (Lead Developer)

Mohamed Abdul Rahim (Intensity Module Contributor)

Suchit (UI Contributor)
