#0311.py
import numpy as np
import cv2

def onMouse(event, x, y, flags, param):
##    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.rectangle(param[0], (x-5, y-5), (x+5, y+5), (255, 0, 0))
        else:
            cv2.circle(param[0], (x, y), 5, (255, 0, 0), 3)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(param[0], (x, y), 5, (0, 0, 255), 3)        
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        param[0] = np.zeros(param[0].shape, np.uint8) + 255   
    cv2.imshow("img", param[0])
    
img = np.zeros((512,512,3), np.uint8) + 255
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse, [img])
cv2.waitKey()
cv2.destroyAllWindows()
