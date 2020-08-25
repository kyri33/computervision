import cv2
import numpy as np

gray_mask = cv2.imread("gray_masked.jpg")
gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_BGR2GRAY)

top_pixels = gray_mask.copy()
print(top_pixels.shape)
for col in range(top_pixels.shape[1]):
    topFound = False
    for row in range(top_pixels.shape[0]):
        if topFound:
            top_pixels[row, col] = 0
        else:
            if top_pixels[row, col]:
                topFound = True

cv2.imshow("top pixels", top_pixels)
cv2.waitKey()
