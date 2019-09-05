# 0512.py
import cv2
import numpy as np
import time
from   matplotlib import pyplot as plt

#1
nPoints = 100000
pts1 = np.zeros((nPoints, 1), dtype=np.uint16)
pts2 = np.zeros((nPoints, 1), dtype=np.uint16)

cv2.setRNGSeed(int(time.time()))
cv2.randn(pts1, mean=(128), stddev=(10))
cv2.randn(pts2, mean=(110), stddev=(20))            

#2
H1 = cv2.calcHist(images=[pts1], channels=[0], mask=None,
                    histSize=[256], ranges=[0, 256])
cv2.normalize(H1, H1, 1, 0, cv2.NORM_L1)
plt.plot(H1, color='r', label='H1')

H2 = cv2.calcHist(images=[pts2], channels=[0], mask=None,
                    histSize=[256], ranges=[0, 256])
cv2.normalize(H2, H2, 1, 0, cv2.NORM_L1)

#3
d1 = cv2.compareHist(H1, H2, cv2.HISTCMP_CORREL)
d2 = cv2.compareHist(H1, H2, cv2.HISTCMP_CHISQR)
d3 = cv2.compareHist(H1, H2, cv2.HISTCMP_INTERSECT)
d4 = cv2.compareHist(H1, H2, cv2.HISTCMP_BHATTACHARYYA)
print('d1(H1, H2, CORREL) =',       d1)
print('d2(H1, H2, CHISQR)=',        d2)
print('d3(H1, H2, INTERSECT)=',     d3)
print('d4(H1, H2, BHATTACHARYYA)=', d4)

plt.plot(H2, color='b', label='H2')
plt.legend(loc='best')
plt.show()
