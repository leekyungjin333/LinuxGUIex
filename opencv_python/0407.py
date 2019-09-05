# 0407.py
import cv2
 
src = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)
roi = cv2.selectROI(src)
print('roi =', roi)

img = src[roi[1]:roi[1]+roi[3],
               roi[0]:roi[0]+roi[2]]

cv2.imshow('Img', img)
cv2.waitKey()
cv2.destroyAllWindows()
