# 0805.py
import cv2
import numpy as np

#1
src = cv2.imread('./data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.01)

#2
res = cv2.dilate(res, None) # 3x3 rect kernel
ret, res = cv2.threshold(res, 0.01*res.max(),255,cv2.THRESH_BINARY)
res8 = np.uint8(res)
cv2.imshow('res8',  res8)

#3
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res8)
print('centroids.shape=', centroids.shape)
print('centroids=',centroids)
centroids = np.float32(centroids)

#4
term_crit=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,10, 0.001)
corners = cv2.cornerSubPix(gray, centroids, (5,5), (-1,-1), term_crit)
print('corners=',corners)

#5 
dst = src.copy()
for x, y in corners[1:]:    
    cv2.circle(dst, (x, y), 5, (0,0,255), 2)

cv2.imshow('dst',  dst) 
cv2.waitKey()
cv2.destroyAllWindows()
