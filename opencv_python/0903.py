# 0903.py
import cv2
import numpy as np
 
src = cv2.imread('./data/chessBoard.jpg')
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#1
mserF = cv2.MSER_create(10)  # cv2.MSER.create(10)
kp= mserF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
cv2.imshow('dst',  dst)

#2
dst2 = dst.copy()
regions, bboxes = mserF.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(dst2, hulls, True, (0, 255, 0))
cv2.imshow('dst2',  dst2)

#3
dst3 = dst.copy()
for i, pts in enumerate(regions):
    box = cv2.fitEllipse(pts)
    cv2.ellipse(dst3, box,  (255,0,0),1)
    x, y, w, h = bboxes[i]
    cv2.rectangle(dst3, (x, y), (x+w, y+h), (0,255,0))     
cv2.imshow('dst3',  dst3)
cv2.waitKey()
cv2.destroyAllWindows()
