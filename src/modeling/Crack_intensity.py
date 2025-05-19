import cv2
# import numpy as np


def process_crack_intensity(binary_img):
    """
    Analyzes a binary preprocessed crack image and returns the image
    with bounding boxes colored based on crack intensity.

    Parameters:
    - binary_img: preprocessed binary image (numpy array, dtype=uint8)

    Returns:
    - visualized_img: image with bounding boxes drawn based on intensity
    """

    def analyze_crack_intensity_v3(
        binary_img, area_thresh_low=0.06, area_thresh_high=0.9, width_thresh=30
    ):
        img_height, img_width = binary_img.shape
        img_area = img_height * img_width

        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        crack_features = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            bbox = binary_img[y : y + h, x : x + w]
            white_pixels = cv2.countNonZero(bbox)
            crack_ratio = white_pixels / img_area
            local_width = min(w, h)

            # Intensity classification logic
            if crack_ratio < area_thresh_low or local_width < width_thresh:
                intensity = 0  # Low
            elif (area_thresh_low <= crack_ratio < area_thresh_high) or (
                local_width < width_thresh * 1.5
            ):
                intensity = 1  # Medium
            else:
                intensity = 2  # High

            crack_features.append({"bbox": (x, y, w, h), "intensity": intensity})

        return crack_features

    def visualize_crack_intensity(img, crack_features):
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for feat in crack_features:
            x, y, w, h = feat["bbox"]
            intensity = feat["intensity"]

            if intensity == 0:
                color = (0, 255, 0)  # Green (Low)
            elif intensity == 1:
                color = (0, 255, 255)  # Yellow (Medium)
            else:
                color = (0, 0, 255)  # Red (High)

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                output,
                f"{intensity}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return output

    crack_features = analyze_crack_intensity_v3(binary_img)
    visualized_img = visualize_crack_intensity(binary_img, crack_features)
    return visualized_img
