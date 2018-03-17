import numpy as np
import cv2
import imutils

EXPECTED_MAX = 100.0
EXPECTED_MIN = -1 * EXPECTED_MAX
FILTER_THRESHOLD = -90.0
MAX_VALUE = 255
MEDIAN_VALUE = MAX_VALUE / 2.0

def normalize_and_filter(data, expected_max=EXPECTED_MAX, median=MEDIAN_VALUE, threshold=FILTER_THRESHOLD):
	data = (data - median) / median * expected_max
	# data[data < threshold] = EXPECTED_MIN
	return data


def zoomin(source, z):
	if z < 1:
		raise ValueError('z must be bigger than 1')
	if z == 1:
		return source

	resized = imutils.resize(source, width=int(round(z * source.shape[1])))	
	top_left = ((resized.shape[0] - source.shape[0]) / 2, (resized.shape[1] - source.shape[1]) / 2)
	cropped = resized[top_left[0]:(top_left[0] + source.shape[0]), top_left[1]:(top_left[1] + source.shape[1])]
	return cropped

img = cv2.imread("batik-parang.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
print(img.shape)
print(gray.shape)
print(color.shape)
cv2.imshow("colored_grayscale", color)
cv2.waitKey(0)

# sift = cv2.xfeatures2d.SIFT_create()
# kp, dsc = sift.detectAndCompute(gray, None)
# print(dsc.shape)

# cv2.imshow("original", img)
# cv2.imshow("rotated 90 degree", imutils.rotate(img, 90))
# cv2.imshow("zoom in 120%", zoomin(img, 1.2))
# cv2.imshow("zoom in 150%", zoomin(img, 1.5))
