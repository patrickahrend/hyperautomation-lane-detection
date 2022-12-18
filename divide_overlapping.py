# Divides the picture into given number of rows and columns
# Here the divided frames are overlapping
# Hard coded right now, have to be changed later

import cv2
import numpy as np
import os

# Load the frame
frame = cv2.imread("test_images/straight_lines1.jpg")
output_dir = "equal_frames/equal_frames_5x1"

# Get the height and width of the frame
height, width = frame.shape[:2]

height = height // 2

num_cols = 5
num_rows = 2

step_x = 128
step_y = 144

rect_height = height - (num_rows - 1) * step_y
rect_width = width - (num_cols - 1) * step_x

# Split the frame into rectangular ROIs
rois = []
for i in range(num_rows):
    for j in range(num_cols):
        x = j * step_x
        y = height + i * step_y
        # Adjust the code so, that the ROIs are overlapping and the frames cover the whole image
        roi = frame[y:y + rect_height, x:x + rect_width]
        rois.append(roi)

# Create a blank image to hold the ROIs
result = np.zeros((height, width, 3), dtype=np.uint8)

# Save the ROIs to separate image files
for i, roi in enumerate(rois):
    cv2.imwrite(f"equal_frames_overlapping_{i}.jpg", roi)
