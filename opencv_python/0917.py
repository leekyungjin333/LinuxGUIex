# 0917.py
import cv2
from   matplotlib import pyplot as plt

src = cv2.imread('./data/people1.png')

#1: HoG in color image
hog1 = cv2.HOGDescriptor()
des1 = hog1.compute(src)
print('des1.shape=', des1.shape)
print('des1=', des1)

#2: HoG in color image
winSize     = (64,128)
blockSize   = (16,16)
blockStride = (8,8)
cellSize    = (8,8)
nbins       = 9
derivAperture = 1
winSigma = -1  # 4.0
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False 
hog2 = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                         derivAperture,winSigma,
                         histogramNormType,L2HysThreshold,
                         gammaCorrection,nlevels, signedGradient)
des2 = hog2.compute(src)
print('des2.shape=', des2.shape)
print('des2=', des2)

#3 HoG in grayscale image
gray  = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
des3 = hog1.compute(gray)
##des3 = hog2.compute(gray)
print('des3.shape=', des3.shape)
print('des3=', des3)

#4
plt.title('HOGDescriptor')
plt.plot(des1[::36], color='b',linewidth=4,label='des1')
plt.plot(des2[::36], color='r',linewidth=2,label='des2')
plt.plot(des3[::36], color='g',linewidth=1,label='des3')
plt.legend(loc='best')
plt.show()
