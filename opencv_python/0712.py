# 0712.py
import cv2
import numpy as np
#1
src = cv2.imread('./data/lena.jpg')

down2 = cv2.pyrDown(src)
down4 = cv2.pyrDown(down2)
print('down2.shape=', down2.shape)
print('down2.shape=', down2.shape)

#2
up2 = cv2.pyrUp(src)
up4 = cv2.pyrUp(up2)
print('up2.shape=', up2.shape)
print('up4.shape=', up4.shape)

cv2.imshow('down2',down2)
##cv2.imshow('down4',down4)
cv2.imshow('up2',up2)
##cv2.imshow('up4',up4)
cv2.waitKey()
cv2.destroyAllWindows()
