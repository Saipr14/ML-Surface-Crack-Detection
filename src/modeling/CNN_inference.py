import mlflow

mlflow.set_experiment("Surface Crack Detection Experiments")
mlflow.set_tracking_uri("https://dagshub.com/Saipr14/ML-Surface-Crack-Detection.mlflow")
client = mlflow.MlflowClient()
models = client.search_registered_models()

print("Registered Models:")
for model in models:
    print(model.name)


# ---------------------------------------------------------------------------------------------

import mlflow.keras  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from preprocessing import preprocess_image  # noqa: E402

# Load trained model
model_name = "Preprocess Custom CNN"
model_version = 1
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.keras.load_model(model_uri)

# intensity_mapping = {0: "Less Intense", 1: "Medium", 2: "High"}

# Load image correctly
img = cv2.imread("../../data/external/Crack/00001.jpg")
if img is None:
    raise ValueError("Error: Could not load image. Check format and path.")
img = preprocess_image(img)
# Convert BGR to RGB (for color images)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (96, 96))

# If grayscale model, convert to grayscale and add 1 channel
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = np.expand_dims(img, axis=-1)

# Normalize
img = img / 255.0

# Reshape for model input
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Show image
plt.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
plt.show()

# Get model predictions
prediction = model.predict(img)
predicted_prob = prediction[0][0]

# Use correct thresholding
predicted_class = 0 if predicted_prob > 0.5 else 1

# Mapping class names
class_mapping = {0: "Non-Crack", 1: "Crack"}
print(
    f"Predicted class: {predicted_class} - {class_mapping[predicted_class]} (Confidence: {predicted_prob:.4f})"
)


# At the top, keep all your imports as-is
"""
import mlflow
import mlflow.keras
import numpy as np
import cv2
from .preprocessing import preprocess_image
def predict_image(img_array):   
    mlflow.set_experiment("Surface Crack Detection Experiments")
    mlflow.set_tracking_uri("https://dagshub.com/Saipr14/ML-Surface-Crack-Detection.mlflow")

    model_name = "Preprocess Custom CNN"
    model_version = 1
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.keras.load_model(model_uri)

    if img_array is None:
        raise ValueError("Image is None.")

    img = preprocess_image(img_array)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_prob = prediction[0][0]
    predicted_class = 0 if predicted_prob > 0.5 else 1
    class_mapping = {0: "Non-Crack", 1: "Crack"}
    return class_mapping[predicted_class], float(predicted_prob)"""
