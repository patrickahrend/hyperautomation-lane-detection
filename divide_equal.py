import cv2
import numpy as np

# Load the frame
frame = cv2.imread("test_images/straight_lines1.jpg")

# Get the height and width of the frame
height, width = frame.shape[:2]

# Calculate the size of each square ROI
roi_size = 80

# Split the frame into equal square parts
rois = []
for i in range(height // roi_size):
    for j in range(width // roi_size):
        x = j * roi_size
        y = i * roi_size
        roi = frame[y:y+roi_size, x:x+roi_size]
        rois.append(roi)

# Create a blank image to hold the ROIs
result = np.zeros((height, width, 3), dtype=np.uint8)

# Draw the ROIs on the result image
for i, roi in enumerate(rois):
    x = (i % (width // roi_size)) * roi_size
    y = int(i / (width // roi_size)) * roi_size
    result[y:y+roi_size, x:x+roi_size] = roi

# Save the ROIs to separate image files
for i, roi in enumerate(rois):
    cv2.imwrite(f"roi_{i}.jpg", roi)

# Display the result
cv2.imshow("Result", result)
cv2.waitKey(0)
