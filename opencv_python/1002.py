# 1002.py
import cv2
import numpy as np

#1
cap = cv2.VideoCapture('./data/vtest.avi')
if (not cap.isOpened()): 
     print('Error opening video')
     
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

TH      = 40  # binary threshold
AREA_TH = 80 # area   threshold 
bkg_gray= cv2.imread('./data/avg_gray.png', cv2.IMREAD_GRAYSCALE)
bkg_bgr = cv2.imread('./data/avg_bgr.png')

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

#2
t = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    t+=1
    print('t =', t)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#2-1 
    diff_gray  = cv2.absdiff(gray, bkg_gray)
##    ret, bImage= cv2.threshold(diff_gray,TH,255,cv2.THRESH_BINARY)
    
#2-2      
    diff_bgr = cv2.absdiff(frame, bkg_bgr)      
    db, dg, dr = cv2.split(diff_bgr)
    ret, bb = cv2.threshold(db,TH,255,cv2.THRESH_BINARY)
    ret, bg = cv2.threshold(dg,TH,255,cv2.THRESH_BINARY)
    ret, br = cv2.threshold(dr,TH,255,cv2.THRESH_BINARY)
 
    bImage = cv2.bitwise_or(bb, bg)
    bImage = cv2.bitwise_or(br, bImage)
      
    bImage = cv2.erode(bImage, None, 5)
    bImage = cv2.dilate(bImage,None, 5)    
    bImage = cv2.erode(bImage, None, 7)

#2-3     
    image, contours, hierarchy = cv2.findContours(bImage, mode, method)
    cv2.drawContours(frame, contours, -1, (255,0,0), 1)   
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > AREA_TH:
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
    
    cv2.imshow('frame',frame)
    cv2.imshow('bImage',bImage)
    cv2.imshow('diff_gray',diff_gray)
    cv2.imshow('diff_bgr',diff_bgr)
    key = cv2.waitKey(25)
    if key == 27:
        break
#3
if cap.isOpened():
    cap.release();
cv2.destroyAllWindows()
