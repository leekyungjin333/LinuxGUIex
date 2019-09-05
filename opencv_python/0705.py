# 0705.py
import cv2
import numpy as np

#1
src1 = cv2.imread('./data/hand.jpg')
hsv1 = cv2.cvtColor(src1, cv2.COLOR_BGR2HSV)
lowerb1 = (0, 40, 0)
upperb1 = (20, 180, 255)
dst1 = cv2.inRange(hsv1, lowerb1, upperb1)

#2
src2 = cv2.imread('./data/flower.jpg')
hsv2 = cv2.cvtColor(src2,cv2.COLOR_BGR2HSV)
lowerb2 = (150, 100, 100)
upperb2 = (180, 255, 255)
dst2 = cv2.inRange(hsv2, lowerb2, upperb2)

#3
cv2.imshow('src1',  src1)
cv2.imshow('dst1',  dst1)
cv2.imshow('src2',  src2)
cv2.imshow('dst2',  dst2)
cv2.waitKey()
cv2.destroyAllWindows()
