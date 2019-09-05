# 0606.py
import cv2
import numpy as np

#1
src  = cv2.imread('./data/rect.jpg', cv2.IMREAD_GRAYSCALE)
#src  = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(src, ksize=(7, 7), sigmaX=0.0)
lap = cv2.Laplacian(blur, cv2.CV_32F,3)

##ret, edge = cv2.threshold(np.abs(lap), 10, 255, cv2.THRESH_BINARY)
##edge = edge.astype(np.uint8)
##cv2.imshow('edge',  edge)

#2
def SGN(x):
    if x >= 0:
        sign = 1
    else:
        sign = -1
    return sign

def zeroCrossing(lap):
    width, height = lap.shape
    Z = np.zeros(lap.shape, dtype=np.uint8)    
    for y in range(1, height-1):
        for x in range(1,width-1):
            neighbors=[lap[y-1,x], lap[y+1,x], lap[y,x-1], lap[y,x+1],
                       lap[y-1,x-1], lap[y-1,x+1], lap[y+1,x-1], lap[y+1,x+1]]                       
            mValue= min(neighbors)
            if SGN(lap[y,x]) != SGN(mValue):
                Z[y, x] = 255
    return Z
edgeZ = zeroCrossing(lap)
cv2.imshow('Zero Crossing',  edgeZ)
cv2.waitKey()    
cv2.destroyAllWindows()
