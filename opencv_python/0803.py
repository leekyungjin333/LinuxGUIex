# 0803.py
import cv2
import numpy as np

#1
src = cv2.imread('./data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
eigen = cv2.cornerMinEigenVal(gray, blockSize=5)
print('eigen.shape=', eigen.shape)

#2
T = 0.2
corners  = np.argwhere(eigen> T)
corners[:,[0, 1]] = corners[:,[1, 0]] # switch x, y
print('len(corners ) =', len(corners ))
dst = src.copy()
for x, y in corners :    
    cv2.circle(dst, (x, y), 3, (0,0,255), 2)
    
cv2.imshow('dst',  dst)
cv2.waitKey()
cv2.destroyAllWindows()
