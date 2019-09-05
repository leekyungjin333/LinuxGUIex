# 0607.py
import cv2
import numpy as np

src  = cv2.imread('./data/rect.jpg', cv2.IMREAD_GRAYSCALE)
##src  = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

#1
kx, ky = cv2.getDerivKernels(1, 0, ksize=3)
sobelX = ky.dot(kx.T)
print('kx=', kx)
print('ky=', ky)
print('sobelX=', sobelX)
gx = cv2.filter2D(src, cv2.CV_32F, sobelX)
##gx = cv2.sepFilter2D(src, cv2.CV_32F, kx, ky)

#2
kx, ky = cv2.getDerivKernels(0, 1, ksize=3)
sobelY = ky.dot(kx.T)
print('kx=', kx)
print('ky=', ky)
print('sobelY=', sobelY)
gy = cv2.filter2D(src, cv2.CV_32F, sobelY)
##gy = cv2.sepFilter2D(src, cv2.CV_32F, kx, ky)

#3
mag   = cv2.magnitude(gx, gy)
ret, edge = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('edge',  edge)
cv2.waitKey()    
cv2.destroyAllWindows()
