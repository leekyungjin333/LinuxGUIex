# 0802.py
import cv2
import numpy as np
#1
src = cv2.imread('./data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.cornerEigenValsAndVecs(gray, blockSize=5, ksize=3)
print('res.shape=', res.shape)
eigen = cv2.split(res)

#2
T = 0.2
ret, edge = cv2.threshold(eigen[0], T, 255, cv2.THRESH_BINARY)
edge = edge.astype(np.uint8)

#3
corners = np.argwhere(eigen[1]>T)
corners[:,[0, 1]] = corners[:,[1, 0]] # switch x, y
print('len(corners) =', len(corners))

dst = src.copy()
for x, y in corners:  
    cv2.circle(dst, (x, y), 5, (0,0,255), 2)
    
cv2.imshow('edge',  edge) 
cv2.imshow('dst',  dst)
cv2.waitKey()
cv2.destroyAllWindows()
