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

step_x = 10
step_y = 5

rect_height = 150
rect_width = 550

num_cols = (width - rect_width) // step_x + 1
num_rows = (height - rect_height) // step_y + 1

# Split the frame into rectangular ROIs
while rect_width < 1200:
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * step_x
            y = height + i * step_y
            roi = frame[y:y + rect_height, x:x + rect_width]
            cv2.imwrite(f"{x}x{y}t{x}x{y + rect_height}t{x + rect_width}x{y + rect_height}t{x + rect_width}x{y}.jpg", roi)
    rect_width = rect_width + 5
    rect_height = rect_height if (rect_height == 200) else (rect_height + 5)
    num_cols = (width - rect_width) // step_x + 1
    num_rows = (height - rect_height) // step_y + 1