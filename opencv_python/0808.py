# 0808.py
import cv2
import numpy as np

#1
src = cv2.imread('./data/circleGrid.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
patternSize = (6, 4)
found, centers = cv2.findCirclesGrid(src, patternSize)
print('centers.shape=', centers.shape)

#2
term_crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
centers2 = cv2.cornerSubPix(gray, centers, (5,5), (-1,-1), term_crit)

#3
dst = src.copy()
cv2.drawChessboardCorners(dst, patternSize, centers2, found)

cv2.imshow('dst',  dst) 
cv2.waitKey()
cv2.destroyAllWindows()
