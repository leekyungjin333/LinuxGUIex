# 0901.py
import cv2
import numpy as np
 
src = cv2.imread('./data/chessBoard.jpg')
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#1
##fastF = cv2.FastFeatureDetector_create()
##fastF =cv2.FastFeatureDetector.create()
fastF =cv2.FastFeatureDetector.create(threshold=30) # 100
kp = fastF.detect(gray) 
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
print('len(kp)=', len(kp))
cv2.imshow('dst',  dst)

#2
fastF.setNonmaxSuppression(False)
kp2 = fastF.detect(gray)
dst2 = cv2.drawKeypoints(src, kp2, None, color=(0,0,255))
print('len(kp2)=', len(kp2))
cv2.imshow('dst2',  dst2)

#3
dst3 = src.copy()
points = cv2.KeyPoint_convert(kp)
for cx, cy in points:
    cv2.circle(dst3, (cx, cy), 3, color=(255, 0, 0), thickness=1)
cv2.imshow('dst3',  dst3)
cv2.waitKey()
cv2.destroyAllWindows()
