#1119.py
# ref: https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html
import numpy as np
import cv2

faceCascade= cv2.CascadeClassifier(
      './haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(
    './haarcascades/haarcascade_eye.xml')

src = cv2.imread('./data/lena.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 3) #(gray, 1.1, 0)

for (x, y, w, h) in faces:
    cv2.rectangle(src, (x,y),(x+w, y+h),(255,0,0), 2)
    
    roi_gray  = gray[y:y+h, x:x+w]
    roi_color = src[y:y+h, x:x+w]
    
    eyes = eyeCascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
