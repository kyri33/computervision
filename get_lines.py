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

for rho, theta in lines[1:, 0]:
	if abs(theta_0 - theta) > 0.4:
		theta_1 = theta
		rho_1 = rho
		break

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

baseline = None
sideline = None
if theta_0 < 1.6:
    sideline = (rho_0, theta_0)
    baseline = (rho_1, theta_1)
else:
    baseline = (rho_0, theta_0)
    sideline = (rho_1, theta_1)

#cv2.imshow('lines', top_lines)
#cv2.waitKey()

THRESH = 35
OFFSET_X = 0.01
OFFSET_Y = 0.2
ANGLE_DIFF = 0.25
ANGLE_DIFF2 = 0.35
DIST_DIFF = 50

parr = lambda theta1, theta2: abs(theta1 - theta2) < ANGLE_DIFF
parr2 = lambda theta1, theta2: abs(theta1 - theta2) < ANGLE_DIFF2
far = lambda rho1, rho2: abs(rho1 - rho2) > DIST_DIFF

flooded = cv2.imread("double_filled.jpg")
flooded_gray = cv2.cvtColor(flooded.copy(), cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(flooded_gray.copy(), 50, 200)
cv2.imshow("canny filled", canny)
#cv2.waitKey()

padded_canny = np.zeros(canny.shape, np.uint8)
y_range_bottom = int(OFFSET_Y * canny.shape[0])
y_range_top = int(0.75 * canny.shape[0])
x_range_left = int(OFFSET_X * canny.shape[1])
x_range_right = int(0.9 * canny.shape[1])

padded_canny[y_range_bottom:y_range_top, x_range_left: x_range_right] = \
        canny[y_range_bottom:y_range_top, x_range_left:x_range_right]

cv2.imshow("padded canny", padded_canny)
#cv2.waitKey()

lines = cv2.HoughLines(padded_canny.copy(), 1, np.pi/180, THRESH)
'''
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    col = (122, 0, 0)
    if theta < 1.0:
        col = (255, 0, 0)
    cv2.line(padded_canny, (x1, y1), (x2, y2), col, 2)

cv2.imshow("linedflooded", padded_canny)
cv2.waitKey()
'''

freethrowline = None
paintline = None
for line in lines[2:]:
    rho, theta = line[0]
    print(theta)
    if freethrowline is None and parr(theta, baseline[1]) and far(rho, baseline[0]):
        freethrowline = line
    if paintline is None and parr2(theta, sideline[1]) and far(rho, sideline[0]) and theta < 1.0:
        paintline = line

'''
for rho, theta in [freethrowline[0], paintline[0]]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a) )
    cv2.line(padded_canny, (x1, y1), (x2, y2), (255, 0, 0), 2)
    print(theta)
cv2.imshow("lined padded", padded_canny)
cv2.waitKey()
'''

bgr_image = cv2.imread("basketball.png")

for rho, theta in [freethrowline[0], paintline[0], baseline, sideline]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(bgr_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

cv2.imshow("lined", bgr_image)
cv2.waitKey()

def get_intersection(line_1, line_2):
    rho_1, theta_1 = line_1
    rho_2, theta_2 = line_2
    a_1 = np.cos(theta_1)
    b_1 = np.sin(theta_1)
    slope_1 = a_1 / -b_1
    intercept_1 = rho_1 / b_1

    a_2 = np.cos(theta_2)
    b_2 = np.sin(theta_2)
    slope_2 = a_2 / -b_2
    intercept_2 = rho_2 / b_2

    x_intersection = (intercept_2 - intercept_1) / (slope_1 - slope_2)
    y_intersection = x_intersection * slope_1 + intercept_1

    return (x_intersection, y_intersection)

freethrowline = freethrowline[0]
paintline = paintline[0]

base_side = get_intersection(baseline, sideline)
base_paint = get_intersection(baseline, paintline)
freethrow_paint = get_intersection(freethrowline, paintline)
freethrow_side = get_intersection(freethrowline, sideline)
padding = 3
td_base_side=np.array([0 + padding, 0 + padding])
td_base_paint = np.array([10 + padding, 0 + padding])
td_freethrow_paint = np.array([10 + padding, 5.79 + padding])
td_freethrow_side = np.array([0 + padding, 5.79 + padding])

scale = 3
court_width = int((28.65 + padding * 2) * scale)
court_height = int((15.24 + padding * 2) * scale)

src_pts = np.array([[base_side[0], base_side[1]], [base_paint[0], base_paint[1]], \
    [freethrow_paint[0], freethrow_paint[1]], [freethrow_side[0], freethrow_side[1]]])
print(td_base_side * scale, td_base_side)
dst_pts = np.array([td_base_side * scale, td_base_paint * scale, \
            td_freethrow_paint * scale, td_freethrow_side * scale])
print(dst_pts)

h, status = cv2.findHomography(src_pts, dst_pts)

td_court = np.zeros((court_height, court_width), np.uint8)
print(td_court.shape)
td_b_s = td_base_side.astype(int)
td_b_p = td_base_paint.astype(int)
td_f_p = td_freethrow_paint.astype(int)
td_f_s = td_freethrow_side.astype(int)
print(td_b_s, td_b_p, td_f_p, td_f_s)
td_court[td_b_s[0] * scale, td_b_s[1] * scale] = 255
td_court[td_b_p[0] * scale, td_b_p[1] * scale] = 255
td_court[td_f_p[0] * scale, td_f_p[1] * scale] = 255
td_court[td_f_s[0] * scale, td_f_s[1] * scale] = 255

cv2.imshow("court", td_court)
cv2.waitKey()
