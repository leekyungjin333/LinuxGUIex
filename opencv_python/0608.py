# 0608.py
import cv2
import numpy as np

src  = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

#1
def logFilter(ksize=7):
    k2 = ksize//2
    sigma = 0.3*(k2-1) + 0.8
    print('sigma=', sigma)
    LoG = np.zeros((ksize, ksize), dtype=np.float32)
    for y in range(-k2, k2+1):
        for x in range(-k2, k2+1):
            g = -(x*x+y*y)/(2.0*sigma**2.0)
            LoG[y+k2, x+k2] = -(1.0+g)*np.exp(g)/(np.pi*sigma**4.0)
    return LoG

#2
kernel = logFilter() #7, 15, 31, 51
LoG = cv2.filter2D(src, cv2.CV_32F, kernel)
cv2.imshow('LoG',  LoG)

#3
def zeroCrossing2(lap, thresh=0.01):
    width, height = lap.shape
    Z = np.zeros(lap.shape, dtype=np.uint8)    
    for y in range(1, height-1):
        for x in range(1,width-1):
            neighbors=[lap[y-1,x],   lap[y+1,x],   lap[y,x-1],   lap[y,x+1],
                       lap[y-1,x-1], lap[y-1,x+1], lap[y+1,x-1], lap[y+1,x+1]]
            pos = 0
            neg = 0
            for value in neighbors:
                if value > thresh:
                    pos += 1
                if value < -thresh:  # value < thresh
                    neg += 1
            if pos > 0 and neg > 0:
                Z[y, x] = 255                        
    return Z
edgeZ = zeroCrossing2(LoG)
cv2.imshow('Zero Crossing2',  edgeZ)
cv2.waitKey()    
cv2.destroyAllWindows()
