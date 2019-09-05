# 0820.py
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
def compute_Haar_feature1(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f1 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, rows-y+1):
                for w in range(1, (cols-x)//2+1):
                    s1 = rectSum(sumImage, (x,  y, w, h))
                    s2 = rectSum(sumImage, (x+w,y, w, h))
                    f1.append([1, x, y, w, h, s1-s2])    
    return f1
def compute_Haar_feature2(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f2 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, (rows-y)//2+1):
                for w in range(1, cols-x+1):
                    s1 = rectSum(sumImage, (x,  y, w, h))
                    s2 = rectSum(sumImage, (x,y+h, w, h))
                    f2.append([2, x, y, w, h, s2-s1])    
    return f2
def compute_Haar_feature3(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f3 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, rows-y+1):
                for w in range(1, (cols-x)//3+1):
                    s1 = rectSum(sumImage, (x,    y, w, h))
                    s2 = rectSum(sumImage, (x+w,  y, w, h))
                    s3 = rectSum(sumImage, (x+2*w,y, w, h))                    
                    f3.append([3, x, y, w, h, s1-s2+s3])    
    return f3
def compute_Haar_feature4(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f4 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, (rows-y)//3+1):
                for w in range(1, cols-x+1):
                    s1 = rectSum(sumImage, (x,  y,   w, h))
                    s2 = rectSum(sumImage, (x,y+h,   w, h))
                    s3 = rectSum(sumImage, (x,y+2*h, w, h))
                    f4.append([4, x, y, w, h, s1-s2+s3])    
    return f4
def compute_Haar_feature5(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f5 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, (rows-y)//2+1):
                for w in range(1, (cols-x)//2+1):
                    s1 = rectSum(sumImage, (x,  y,   w, h))
                    s2 = rectSum(sumImage, (x+w,y,   w, h))
                    s3 = rectSum(sumImage, (x,  y+h, w, h))
                    s4 = rectSum(sumImage, (x+w,y+h, w, h))
                    f5.append([5, x, y, w, h, s1-s2-s3+s4])    
    return f5

#2
gray = cv2.imread('./data/lenaFace24.jpg', cv2.IMREAD_GRAYSCALE) # 24 x 24
gray_sum = cv2.integral(gray)
f1 = compute_Haar_feature1(gray_sum)
n1 = len(f1)
print('len(f1)=',n1)
for i, a in enumerate(f1[:2]):
    print('f1[{}]={}'.format(i, a))
#3
f2 = compute_Haar_feature2(gray_sum)
n2 = len(f2)
print('len(f2)=',n2)
for i, a in enumerate(f2[:2]):
    print('f2[{}]={}'.format(i, a))

#4
f3 = compute_Haar_feature3(gray_sum)
n3 = len(f3)
print('len(f3)=',n3)
for i, a in enumerate(f3[:2]):
    print('f3[{}]={}'.format(i, a))

#5
f4 = compute_Haar_feature4(gray_sum)
n4 = len(f4)
print('len(f4)=',n4)
for i, a in enumerate(f4[:2]):
    print('f4[{}]={}'.format(i, a))
#6
f5 = compute_Haar_feature5(gray_sum)
n5 = len(f5)
print('len(f5)=',n5)
for i, a in enumerate(f5[:2]):
    print('f5[{}]={}'.format(i, a))
    
print('total features =', n1+n2+n3+n4+n5)
