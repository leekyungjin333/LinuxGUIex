# 0819.py
import cv2
import numpy as np
#1
def rectSum(sumImage, rect):
    x, y, w, h = rect
    a = sumImage[y, x]
    b = sumImage[y, x+w]
    c = sumImage[y+h, x]
    d = sumImage[y+h, x+w]
    return a + d - b - c
#2
def compute_Haar_feature1(sumImage, rect):
    x, y, w, h = rect
##    print(x, y, w, h)
    s1 = rectSum(sumImage, (x,  y, w, h))
    s2 = rectSum(sumImage, (x+w,y, w, h))
##    print('s1=', s1)
##    print('s2=', s2)    
    return s1-s2
def compute_Haar_feature2(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x,  y, w, h))
    s2 = rectSum(sumImage, (x,y+h, w, h)) 
    return s2-s1
def compute_Haar_feature3(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x,    y, w, h))
    s2 = rectSum(sumImage, (x+w,  y, w, h))
    s3 = rectSum(sumImage, (x+2*w,y, w, h))
    return s1-s2+s3
def compute_Haar_feature4(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x,    y,  w, h))
    s2 = rectSum(sumImage, (x,  y+h,  w, h))
    s3 = rectSum(sumImage, (x,  y+2*h,w, h))
    return s1-s2+s3
def compute_Haar_feature5(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x,    y, w, h))
    s2 = rectSum(sumImage, (x+w,  y, w, h))
    s3 = rectSum(sumImage, (x,  y+h, w, h))
    s4 = rectSum(sumImage, (x+w,y+h, w, h))
    return s1+s4-s2-s3

#3
A = np.arange(1, 6*6+1).reshape(6, 6).astype(np.uint8)
print('A=', A)

h, w = A.shape
sumA = cv2.integral(A)
print('sumA=', sumA)

#4
f1 = compute_Haar_feature1(sumA, (0, 0, w//2, h))    # 3, 6
print('f1=', f1)

#5
f2 = compute_Haar_feature2(sumA, (0, 0, w, h//2))    # 6, 3
print('f2=', f2)

#6
f3 = compute_Haar_feature3(sumA, (0, 0, w//3, h))    # 2, 6
print('f3=', f3)

#7
f4 = compute_Haar_feature4(sumA, (0, 0, w, h//3))    # 6, 2
print('f4=', f4)

#8
f5 = compute_Haar_feature5(sumA, (0, 0, w//2, h//2)) # 3, 3
print('f5=', f5)
