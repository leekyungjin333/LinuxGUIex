# 0905.py
import cv2
import numpy as np
 
src = cv2.imread('./data/chessBoard.jpg')
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#1
##goodF = cv2.GFTTDetector.create()
goodF = cv2.GFTTDetector_create()
kp= goodF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))   
cv2.imshow('dst',  dst)

#2
goodF2 = cv2.GFTTDetector_create(maxCorners= 50,
                                qualityLevel=0.1,
                                minDistance = 10,
                                useHarrisDetector=True)
kp2= goodF2.detect(gray)
print('len(kp2)=', len(kp2))
dst2 = cv2.drawKeypoints(gray, kp2, None, color=(0,0,255))   
cv2.imshow('dst2',  dst2)
cv2.waitKey()
cv2.destroyAllWindows()
