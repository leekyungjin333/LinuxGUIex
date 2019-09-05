#0301.py

import cv2
import numpy as np

# White 배경 배열 생성
# shpae=(512,512,3)    512 X 512 크기의 3채널 컬러 영상
# dtype = np.uint8     영상화소가 부호없는 8비트 정수
# np.zeros( ) + 255    영상의 모든 채널 값이 255로 흰색이다.
img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255

#img = np.ones((512,512,3), np.uint8) * 255
#img = np.full((512,512,3), (255, 255, 255), dtype= np.uint8)
#img = np.zeros((512,512, 3), np.uint8) # Black 배경
pt1 = 100, 100
pt2 = 400, 400
cv2.rectangle(img, pt1, pt2, (255, 0, 0), -1)

cv2.line(img, (0, 0), (500, 0), (255, 0, 0), 5)
cv2.line(img, (0, 0), (0, 500), (0,0,255), 5)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
