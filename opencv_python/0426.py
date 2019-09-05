# 0426.py
import cv2
import numpy as np

src = cv2.imread('./data/lena.jpg') 
b, g, r = cv2.split(src) 
cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)

X = src.reshape(-1, 3)
print('X.shape=', X.shape)

mean, eVects = cv2.PCACompute(X, mean=None)
print('mean = ', mean)
print('eVects = ', eVects)

Y =cv2.PCAProject(X, mean, eVects)
Y = Y.reshape(src.shape)
print('Y.shape=', Y.shape)

eImage = cv2.split(Y)
for i in range(3):
    cv2.normalize(eImage[i], eImage[i], 0, 255, cv2.NORM_MINMAX)
    eImage[i]=eImage[i].astype(np.uint8)
    
cv2.imshow('eImage[0]', eImage[0])
cv2.imshow('eImage[1]', eImage[1])
cv2.imshow('eImage[2]', eImage[2])
cv2.waitKey()    
cv2.destroyAllWindows()
