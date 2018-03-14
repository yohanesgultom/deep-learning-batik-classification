import numpy as np
import cv2
import imutils

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
print(img.shape)
cv2.imshow("original", img)
cv2.imshow("rotated 90 degree", imutils.rotate(img, 90))
cv2.imshow("zoom in 120%", zoomin(img, 1.2))
cv2.imshow("zoom in 150%", zoomin(img, 1.5))
cv2.waitKey(0)