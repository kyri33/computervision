import cv2
import numpy as np

gray_mask = cv2.imread("gray_masked.jpg")
gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_BGR2GRAY)

top_pixels = gray_mask.copy()
for col in range(top_pixels.shape[1]):
    topFound = False
    for row in range(top_pixels.shape[0]):
        if topFound:
            top_pixels[row, col] = 0
        else:
            if top_pixels[row, col]:
                topFound = True

cv2.imshow("top pixels", top_pixels)
#cv2.waitKey()

top_lines = top_pixels.copy()

lines = cv2.HoughLines(top_lines.copy(), 5, np.pi/180 * 3, 75)

# SHOW LINES 
'''
for line in lines:
	for rho, theta in line:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
	cv2.line(top_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
'''

theta_0 = lines[0][0][1]
rho_0 = lines[0][0][0]
theta_1 = None
rho_1 = None
print(lines[1:, 0])
for rho, theta in lines[1:, 0]:
	print(abs(theta_0 - theta))
	if abs(theta_0 - theta) > 0.4:
		theta_1 = theta
		rho_1 = rho
		break
print(rho_1, theta_1)
for rho, theta in [[rho_0, theta_0], [rho_1, theta_1]]:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(top_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('lines', top_lines)
cv2.waitKey()

THRESH = 50
OFFSET_X = 0.01
OFFSET_Y = 0.2
ANGLE_DIFF = 0.25
ANGLE_DIFF2 = 0.35
DIST_DIFF = 50

parr = lambda theta1, theta2: abs(theta1 - theta2) < ANGLE_DIFF
parr2 = lambda theta1, theta2: abs(theta1 - theta2) < ANGLE_DIFF2
far = lambda rho1, rho2: abs(rho1 - rho2) > DIST_DIFF

