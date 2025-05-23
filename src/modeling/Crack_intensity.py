import cv2


def process_crack_intensity(binary_img):
    def scale_thresholds(img_width, img_height, base_img_size=(227, 227)):
        base_area = base_img_size[0] * base_img_size[1]
        new_area = img_width * img_height
        area_scale = base_area / new_area
        width_scale = img_width / base_img_size[0]

        return {
            "area_thresh_low": 0.078 * area_scale,
            "area_thresh_high": 0.106 * area_scale,
            "width_thresh": int(40 * width_scale),
        }

    # Function to analyze crack intensity
    def analyze_crack_intensity_v3(binary_img, thresholds):
        img_height, img_width = binary_img.shape
        img_area = img_height * img_width
        print(f"Image Size: {img_height} x {img_width}")
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
            print(f"Crack Ratio: {crack_ratio:.4f}")
            print(
                f"Crack ratio thresholds: {thresholds['area_thresh_low']:.4f} - {thresholds['area_thresh_high']:.4f}"
            )
            local_width = min(w, h)
            print(f"Local Width: {local_width}")
            print(f"Local width thersold: {thresholds['width_thresh']}")
            print(f"Area: {area}")
            # Intensity classification logic
            if (
                crack_ratio < thresholds["area_thresh_low"]
                or local_width < thresholds["width_thresh"]
            ):
                intensity = 0  # Low
            elif (
                thresholds["area_thresh_low"]
                <= crack_ratio
                < thresholds["area_thresh_high"]
            ) or (
                local_width > thresholds["width_thresh"]
                and local_width < thresholds["width_thresh"] * 2.5
            ):
                intensity = 1  # Medium
            else:
                intensity = 2  # High
            print(f"Intensity: {intensity}")
            crack_features.append({"bbox": (x, y, w, h), "intensity": intensity})

        return crack_features

    def visualize_crack_intensity(img, crack_features):
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for feat in crack_features:
            intensity = max((feat["intensity"] for feat in crack_features), default=0)

            if intensity == 0:
                color = (0, 255, 0)  # Green (Low)
            elif intensity == 1:
                color = (0, 255, 255)  # Yellow (Medium)
            else:
                color = (0, 0, 255)  # Red (High)

            h, w = img.shape
            cv2.rectangle(output, (0, 0), (w - 1, h - 1), color, 4)
            cv2.putText(
                output,
                f"{'Low' if intensity == 0 else 'Medium' if intensity == 1 else 'High'}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        return output

    # Step 1: Scale thresholds based on input image size
    img_height, img_width = binary_img.shape
    thresholds = scale_thresholds(img_width, img_height)

    # Step 2: Analyze crack intensities
    crack_features = analyze_crack_intensity_v3(binary_img, thresholds)

    # Step 3: Visualize the results
    visualized_img = visualize_crack_intensity(binary_img, crack_features)

    return visualized_img
