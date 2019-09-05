# 1012.py
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

cap = cv2.VideoCapture('./data/ball.wmv')
if (not cap.isOpened()): 
     print('Error opening video')    
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
roi_mask   = np.zeros((height, width), dtype=np.uint8)
term_crit = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,10, 1)


#3: Kalman Filter setup
q = 1e-5   #  process noise covariance
r = 0.01 #  measurement noise covariance, r = 1
dt = 1
KF = cv2.KalmanFilter(4,2,0)
KF.transitionMatrix = np.array([[1,0,dt,0],         
                                [0,1,0,dt],
                                [0,0,1,0],
                                [0,0,0,1]], np.float32)  # A
KF.measurementMatrix = np.array([[1,0,0,0],
                                 [0,1,0,0]],np.float32)  # H
#4 
t = 0
while True:
     ret, frame = cap.read()
     if not ret: break
     t+=1
     print('t=',t)
     frame2 = frame.copy() # camShift
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     mask = cv2.inRange(hsv,(0, 60, 32),(180,255,255))

     if mouse_status==2:
          x1, y1, x2, y2 = roi
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
     if mouse_status==3:
          print('initialize....')
          mouse_status = 0
          x1, y1, x2, y2 = roi          
          mask_roi = mask[y1:y2, x1:x2]
          hsv_roi  =  hsv[y1:y2, x1:x2]
          
          hist_roi = cv2.calcHist([hsv_roi],[0],mask_roi,[16],[0,180])

          cv2.normalize(hist_roi,hist_roi,0,255,cv2.NORM_MINMAX)
          H1 = hist_roi.copy()
          cv2.normalize(H1,H1,0.0,1.0,cv2.NORM_MINMAX)          
          track_window = (x1, y1, x2-x1, y2-y1) # meanShift
          
#4-1: Kalman filter initialize    
          KF.processNoiseCov     = q* np.eye(4, dtype=np.float32)   # Q         
          KF.measurementNoiseCov = r* np.eye(2, dtype=np.float32) # R
          KF.errorCovPost  = np.eye(4, dtype=np.float32)             # P0 = I

          x, y, w, h = track_window
          cx = x + w/2
          cy = y + h/2
          KF.statePost=np.array([[cx],[cy],[0.],[0.]],dtype=np.float32)
          tracking_start = True
               
     if tracking_start:
#4-2
          predict  = KF.predict()

#4-3: CamShift tracking
          backP = cv2.calcBackProject([hsv],[0],hist_roi,[0,180],1)
          backP &= mask
         
          track_box,track_window=cv2.CamShift(backP, track_window, term_crit)

          cv2.ellipse(frame, track_box, (0, 0, 255), 2)
          cx, cy = track_box[0]
          cv2.circle(frame, (round(cx),round(cy)), 5, (0,0,255), -1)
##          pts = cv2.boxPoints(track_box)
##          pts = np.int0(pts) # np.int32
##          dst = cv2.polylines(frame,[pts],True, (0, 0, 255),2)

#4-4: Kalman correct
          z = np.array([[cx],[cy]],dtype=np.float32) # measurement
          estimate = KF.correct(z)
          estimate = np.int0(estimate)          
#4-5         
          cx2, cy2 = estimate[0][0], estimate[1][0]  
          track_box2 = ((cx2, cy2), track_box[1], track_box[2])
          cv2.ellipse(frame, track_box2, (255, 0, 0), 2)          
          cv2.circle(frame, (cx2,cy2), 5, (255,0,0), -1)
          
     cv2.imshow('tracking',frame)
     key = cv2.waitKey(25)
     if key == 27:
          break
if cap.isOpened():
     cap.release();
cv2.destroyAllWindows()
