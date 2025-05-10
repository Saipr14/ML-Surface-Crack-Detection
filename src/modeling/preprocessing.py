#!/usr/bin/env python
# coding: utf-8

# This NoteBook is for Preprocessing the External Image Dataset

# Importing packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def percolation_fill(img, gap_threshold=2):
    h, w = img.shape
    visited = np.zeros_like(img, dtype=bool)
    result = np.zeros_like(img)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def is_valid(y, x):
        return 0 <= y < h and 0 <= x < w

    def bfs(y, x):
        q = deque()
        q.append((y, x))
        visited[y, x] = True

        while q:
            cy, cx = q.popleft()
            result[cy, cx] = 255

            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if is_valid(ny, nx) and not visited[ny, nx]:
                    if img[ny, nx] > 0:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                    else:
                        # Try to bridge the gap
                        for g in range(1, gap_threshold + 1):
                            gy, gx = cy + dy * g, cx + dx * g
                            if (
                                is_valid(gy, gx)
                                and img[gy, gx] > 0
                                and not visited[gy, gx]
                            ):
                                # Fill intermediate gap
                                for i in range(g + 1):
                                    fy, fx = cy + dy * i, cx + dx * i
                                    if is_valid(fy, fx):
                                        result[fy, fx] = 255
                                        visited[fy, fx] = True
                                q.append((gy, gx))
                                break

    for y in range(h):
        for x in range(w):
            if img[y, x] > 0 and not visited[y, x]:
                bfs(y, x)

    return result


def smart_percolation_fill(img):
    # Stage 1: conservative fill to preserve thin crack detail
    conservative_fill = percolation_fill(img, gap_threshold=15)

    # Stage 2: aggressive fill for longer cracks (connect major gaps)
    aggressive_fill = percolation_fill(conservative_fill, gap_threshold=20)

    # Merge both with OR (preserves both detail and coverage)
    fine_made = cv2.bitwise_or(conservative_fill, aggressive_fill)

    return fine_made


def remove_small_components(img, min_area=10):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(img)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 255, thickness=cv2.FILLED)
    return result


def enhance_contrast_clahe(gray_img):
    return


def preprocess_image(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Optional: histogram equalization instead of CLAHE
    img = cv2.equalizeHist(img_gray)

    blurred = cv2.bilateralFilter(img, 9, 75, 75)

    v = np.median(blurred)
    lower = max(0, int(0.66 * v))
    upper = min(255, int(1.33 * v))

    edges = cv2.Canny(blurred, lower, upper)

    kernel = np.ones((2, 2), np.uint8)
    filled_edges = smart_percolation_fill(edges)
    opened = cv2.morphologyEx(filled_edges, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = remove_small_components(opened, min_area=10)
    return cleaned


# Function to plot the Image
def plot_images(images, category):
    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"{category} Image {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
