#1120.py
'''
 pip install youtube_dl
 pip install pafy
'''
import numpy as np
import cv2, pafy

faceCascade= cv2.CascadeClassifier(
      './haarcascades/haarcascade_frontalface_default.xml')

url = 'https://www.youtube.com/watch?v=S_0ikqqccJs'
video = pafy.new(url)
print('title = ', video.title)

best = video.getbest(preftype='webm')
print('best.resolution', best.resolution)

cap=cv2.VideoCapture(best.url)
while(True):
        retval, frame = cap.read()
        if not retval:
                break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(gray) #minSize=(50, 50)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,0,0), 2)           
        cv2.imshow('frame',frame)
 
        key = cv2.waitKey(25)
        if key == 27: # Esc
                break
cv2.destroyAllWindows()
