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


