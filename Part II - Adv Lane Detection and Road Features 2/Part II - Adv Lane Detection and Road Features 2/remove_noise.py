import cv2
import numpy as np

# load image
img = cv2.imread("testing1.jpg")
def denoise(img):
    noiseless_image_bw = cv2.fastNlMeansDenoising(img, None, 100, 7, 21)
    cv2.imwrite("method.jpg", noiseless_image_bw)
    return noiseless_image_bw