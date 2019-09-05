# 0817.py
import cv2
import numpy as np

#1
A = np.arange(1, 17).reshape(4, 4).astype(np.uint8)
print('A=', A)

#2
sumA, sqsumA, tiltedA = cv2.integral3(A)
print('sumA=', sumA)
print('sqsumA=', np.uint32(sqsumA))
print('tiltedA=', tiltedA)
