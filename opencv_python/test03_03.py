import cv2
import numpy as np

img = np.zeros(shape=(510,510,3), dtype=np.float)
layer1 = np.zeros(shape=(510,510,3), dtype=np.float)
layer2 = np.zeros(shape=(510,510,3), dtype=np.float)
layer3 = np.zeros(shape=(510,510,3), dtype=np.float)

cx1 = img.shape[0]//2
cy1 = img.shape[1]//3

cx2 = img.shape[0]//3
cy2 = (img.shape[1]//3)*2

cx3 = (img.shape[0]//3)*2
cy3 = (img.shape[0]//3)*2

cv2.circle(layer1, (cx1, cy1), 120, color=(0, 0, 255), thickness=-1)
cv2.circle(layer2, (cx2, cy2), 120, color=(0, 255, 0), thickness=-1)
cv2.circle(layer3, (cx3, cy3), 120, color=(255, 0, 0), thickness=-1)

img = cv2.add(layer1, layer2)
img = cv2.add(img, layer3)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()