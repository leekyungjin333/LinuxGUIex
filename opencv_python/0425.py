# 0425.py
import cv2
import numpy as np

X = np.array([[0, 0,  0,100,100,150, -100,-150],
                 [0,50,-50,  0, 30,100,  -20,-100]], dtype=np.float64)
X = X.transpose() # X = X.T

##mean = cv2.reduce(X, 0, cv2.REDUCE_AVG)
##print('mean = ', mean)

mean, eVects = cv2.PCACompute(X, mean=None)
print('mean = ', mean)
print('eVects = ', eVects)

Y =cv2.PCAProject(X, mean, eVects)
print('Y = ', Y)

X2 =cv2.PCABackProject(Y, mean, eVects)
print('X2 = ', X2)
print(np.allclose(X, X2))
cv2.waitKey()    
cv2.destroyAllWindows()
