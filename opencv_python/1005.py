# 1005.py
import cv2
import numpy as np

#1
roi  = None
drag_start = None
mouse_status = 0
tracking_start  = False
def onMouse(event, x, y, flags, param=None):
     global roi
     global drag_start
     global mouse_status
     global tracking_start   
     if event == cv2.EVENT_LBUTTONDOWN:
          drag_start = (x, y)
          mouse_status = 1
          tracking_start = False
     elif event == cv2.EVENT_MOUSEMOVE:
          if flags == cv2.EVENT_FLAG_LBUTTON:
               xmin = min(x, drag_start[0])
               ymin = min(y, drag_start[1])
               xmax = max(x, drag_start[0])
               ymax = max(y, drag_start[1])
               roi = (xmin, ymin, xmax, ymax)
               mouse_status = 2 # dragging
     elif event == cv2.EVENT_LBUTTONUP:
          mouse_status = 3 # complete

#2          
cv2.namedWindow('tracking')
cv2.setMouseCallback('tracking', onMouse)

cap = cv2.VideoCapture('./data/checkBoard3x3.avi')
if (not cap.isOpened()): 
     print('Error opening video')
     
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
roi_mask   = np.zeros((height, width), dtype=np.uint8)

params = dict(maxCorners=16,qualityLevel=0.001,minDistance=10,blockSize=5)
term_crit = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,10,0.01)
params2 = dict(winSize= (5,5), maxLevel = 3, criteria =  term_crit)

#3 
t = 0
while True:
     ret, frame = cap.read()
     if not ret: break
     t+=1
     print('t=',t)
     imgC = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     imgC = cv2.GaussianBlur(imgC, (5, 5), 0.5)
#3-1
     if mouse_status==2:
          x1, y1, x2, y2 = roi
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#3-2          
     if mouse_status==3:
          print('initialize....')
          mouse_status = 0
          x1, y1, x2, y2 = roi
          roi_mask[:,:] = 0
          roi_mask[y1:y2, x1:x2] = 1
          p1 = cv2.goodFeaturesToTrack(imgC,mask=roi_mask,**params)
          if len(p1)>=4:
               p1 = cv2.cornerSubPix(imgC, p1, (5,5),(-1,-1), term_crit)
               rect = cv2.minAreaRect(p1)
               box_pts = cv2.boxPoints(rect).reshape(-1,1,2)
               tracking_start = True
#3-3               
     if tracking_start:
          p2,st,err= cv2.calcOpticalFlowPyrLK(imgP,imgC,p1,None,**params2)
          p1r,st,err=cv2.calcOpticalFlowPyrLK(imgC,imgP,p2,None,**params2)
          d = abs(p1-p1r).reshape(-1, 2).max(-1)
          stat = d < 1.0  # 1.0 is distance threshold
          good_p2 = p2[stat==1].copy()
          good_p1 = p1[stat==1].copy()
          for x, y in good_p2.reshape(-1, 2):
               cv2.circle(frame, (x, y), 3, (0,0,255), -1)

          if len(good_p2)<4:
               continue
          H, mask = cv2.findHomography(good_p1, good_p2, cv2.RANSAC, 3.0)
          box_pts = cv2.perspectiveTransform(box_pts, H)
          cv2.polylines(frame,[np.int32(box_pts)],True,(255,0, 0),2)
          p1 = good_p2.reshape(-1,1,2)

#3-4
     cv2.imshow('tracking',frame)
     imgP = imgC.copy()
     key = cv2.waitKey(25)
     if key == 27:
          break
if cap.isOpened():
     cap.release();
cv2.destroyAllWindows()
