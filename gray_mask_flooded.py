import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from collections import deque

np.set_printoptions(threshold=np.inf)
bgr_image = cv2.imread("basketball.png")

# Get Dominant Colors

dom_thresh = 0.02
ycr_image = cv2.cvtColor(bgr_image.copy(), cv2.COLOR_BGR2YCR_CB)
#cv2.imshow('ycr', ycr_image)
#cv2.waitKey()
hist = cv2.calcHist([ycr_image], [1, 2], None, [256,256], [0, 256, 0, 256])

# Plot histogram
'''
plt.plot(hist)
plt.show()
'''

peak1_flat_idx = np.argmax(hist)
peak1_idx = np.unravel_index(peak1_flat_idx, hist.shape)

thresh = 0.02
connected_hist = set()
sum_val = 0
subtracted_hist = np.copy(hist)
min_passing_val = thresh * hist[peak1_idx]

connected_hist.add(peak1_idx)
sum_val += hist[peak1_idx]
subtracted_hist[peak1_idx] = 0
queue = deque([peak1_idx])
while queue:
    x, y = queue.popleft()
    toAdd = []
    
    if x > 1:
        toAdd.append((x - 1, y))
    if x < hist.shape[0] - 1:
        toAdd.append((x + 1, y))
    if y > 1:
        toAdd.append((x, y - 1))
    if y < hist.shape[1] - 1:
        toAdd.append((x, y + 1))
    
    for idx in toAdd:
        if idx not in connected_hist and hist[idx] >= min_passing_val:
            connected_hist.add(idx)
            subtracted_hist[idx] = 0
            sum_val += hist[idx]
            queue.append(idx)

# Create Gray Mask from dominant connected history

YCBCR_BLACK = (0, 128, 128)
YCBCR_WHITE = (255, 128, 128)
yc_image = cv2.cvtColor(bgr_image.copy(), cv2.COLOR_BGR2YCR_CB)
for row in range(yc_image.shape[0]):
	for col in range(yc_image.shape[1]):
		idx = (row, col)
		_, cr, cb = yc_image[idx]
		if (cr, cb) not in connected_hist:
			yc_image[idx] = YCBCR_BLACK
		else:
			yc_image[idx] = YCBCR_WHITE

bgr_masked = cv2.cvtColor(yc_image.copy(), cv2.COLOR_YCR_CB2BGR)
gray_masked = cv2.cvtColor(bgr_masked.copy(), cv2.COLOR_BGR2GRAY)
cv2.imshow("masked", gray_masked)
#cv2.waitKey()

# Flood holes of mask with contour filling


def fill_holes_with_contour_filling(gray_mask, inverse=False):

	filled = gray_mask.copy()
	if inverse:
		filled = cv2.bitwise_not(filled)
	contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		cv2.drawContours(filled, [cnt], 0, 255, -1)
	if inverse:
		filled = cv2.bitwise_not(filled)

	cv2.imshow("filled", filled)
	cv2.waitKey()
	return filled

filled = fill_holes_with_contour_filling(gray_masked)
filled2 = fill_holes_with_contour_filling(filled, inverse=True)
