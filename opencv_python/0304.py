#0304.py
import cv2
import numpy as np

img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
ptCenter = img.shape[0]//2, img.shape[1]//2

#타원 가로지름:200, 세로지름:100
size = 200,100

cv2.ellipse(img, ptCenter, size, 0, 0, 360, (255, 0, 0))
cv2.ellipse(img, ptCenter, size, 45, 0, 360, (0, 0, 255))

box = (ptCenter, size, 0)
cv2.ellipse(img, box,  (255, 0, 0), 5)

box = (ptCenter, size, 45)
cv2.ellipse(img, box,  (0, 0, 255), 5)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
