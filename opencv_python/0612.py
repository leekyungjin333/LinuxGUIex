# 0612.py
import cv2
import numpy as np

src   = cv2.imread('./data/alphabet.bmp', cv2.IMREAD_GRAYSCALE)
tmp_A   = cv2.imread('./data/A.bmp', cv2.IMREAD_GRAYSCALE)
tmp_S   = cv2.imread('./data/S.bmp', cv2.IMREAD_GRAYSCALE)
tmp_b   = cv2.imread('./data/b.bmp', cv2.IMREAD_GRAYSCALE)
dst  = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)  # 출력 표시 영상

#1
R1 = cv2.matchTemplate(src, tmp_A, cv2.TM_SQDIFF_NORMED)
minVal, _, minLoc, _ = cv2.minMaxLoc(R1)
print('TM_SQDIFF_NORMED:', minVal, minLoc)

w, h = tmp_A.shape[:2]
cv2.rectangle(dst, minLoc, (minLoc[0]+h, minLoc[1]+w), (255, 0, 0), 2)

#2
R2 = cv2.matchTemplate(src, tmp_S, cv2.TM_CCORR_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R2)
print('TM_CCORR_NORMED:', maxVal, maxLoc)
w, h = tmp_S.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0]+h, maxLoc[1]+w), (0, 255, 0), 2)

#3
R3 = cv2.matchTemplate(src, tmp_b, cv2.TM_CCOEFF_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R3)
print('TM_CCOEFF_NORMED:', maxVal, maxLoc)
w, h = tmp_b.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0]+h, maxLoc[1]+w), (0, 0, 255), 2)

cv2.imshow('dst',  dst)
cv2.waitKey()
cv2.destroyAllWindows()
