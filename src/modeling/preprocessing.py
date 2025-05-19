#!/usr/bin/env python
# coding: utf-8

# This NoteBook is for Preprocessing the External Image Dataset

# Importing packages
import cv2
import numpy as np
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
    aggressive_fill = percolation_fill(conservative_fill, gap_threshold=18)

    # Merge both with OR (preserves both detail and coverage)
    fine_made = cv2.bitwise_or(conservative_fill, aggressive_fill)

    return fine_made


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 1)
    blurred = cv2.bilateralFilter(blurred, 9, 15, 15)
    # Use median to find optimal Canny thresholds
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v)) - 90
    upper = int(min(255, 1.33 * v)) + 15
    edges = cv2.Canny(blurred, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    smoothened_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Use custom percolation fill to connect broken cracks
    filled_edges = smart_percolation_fill(smoothened_edges)
    morph_kernel = np.ones((3, 3), np.uint8)
    filled_edges = cv2.morphologyEx(
        filled_edges, cv2.MORPH_CLOSE, morph_kernel, iterations=5
    )

    return filled_edges
