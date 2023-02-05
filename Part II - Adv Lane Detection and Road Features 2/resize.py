## Written by Katya/Aral/Jasmina
# Takes pictures from a file and resizes them to a given size (1280x720 in our case)
# and saves them in a new folder
# This is necessary because the algorithm only works for pictures of a certain size

import cv2
import os

# Set the input and output directories
input_dir = "test"
output_dir = "test_resized"

# Iterate over the files in the input directory
for file in os.listdir(input_dir):
    # Read the image
    image = cv2.imread(os.path.join(input_dir, file))

    # Resize the image
    resized_image = cv2.resize(image, (1280, 720))

    # Save the resized image
    cv2.imwrite(os.path.join(output_dir, file), resized_image)
