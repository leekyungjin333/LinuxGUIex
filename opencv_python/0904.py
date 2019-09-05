# 0904.py
import cv2
import numpy as np
 
src = cv2.imread('./data/chessBoard.jpg')
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

#1
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.thresholdStep = 5
params.minThreshold = 20
params.maxThreshold = 100
params.minDistBetweenBlobs = 5
params.filterByArea = True
params.minArea = 25
params.maxArea = 5000
params.filterByConvexity = True
params.minConvexity = 0.89

#2
##blobF = cv2.SimpleBlobDetector.create(params)
##blobF = cv2.SimpleBlobDetector_create(params)
blobF = cv2.SimpleBlobDetector_create()
kp= blobF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))

#3
for f in kp:
    r = int(f.size/2)
    cx, cy = f.pt
    cv2.circle(dst, (round(cx),round(cy)),r,(0,0,255),2)
    
cv2.imshow('dst',  dst)
cv2.waitKey()
cv2.destroyAllWindows()
