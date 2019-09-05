# 1008.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''ref: Greg Welch and Gary Bishop, 'An Introduction to the Kalman Filter', 2006.
        Estimating a Random Constant : off-line with cv2.KalmanFilter'''
#1
x = -0.37727 # the truth value

q = 1e-5    #  process noise covariance
r = 0.01    #  measurement noise covariance, 1, 0.0001

KF = cv2.KalmanFilter(1, 1, 0)             # B = 0
KF.transitionMatrix      = np.ones((1, 1)) # A = 1
KF.measurementMatrix     = np.ones((1, 1)) # H = 1
KF.processNoiseCov       = q * np.eye(1)   # Q
KF.measurementNoiseCov   = r * np.eye(1)   # R

#2 initial value
KF.errorCovPost        = np.ones((1, 1))  # P0 = 1
KF.statePost           = np.zeros((1, 1)) # x0 = 0

N = 50
z = np.random.randn(N, 1)*np.sqrt(r) + x  # measurement
X = [KF.statePost[0,0]]        # initial value
P = [KF.errorCovPost[0,0]]     # initial errorCovPost

#3
for k in range(1, N):
     predict = KF.predict() 
     estimate = KF.correct(z[k])
     X.append(estimate[0,0])        # KF.statePost[0,0]
     P.append(KF.errorCovPost[0,0])
#4     
plt.figure(1)      
plt.xlabel('k')
plt.ylabel('X(k)')
plt.axis([0, N, x-3*np.sqrt(r), x+3*np.sqrt(r)])
plt.plot([0, N], [x, x], 'g-')  # the truth value line

plt.plot(X, 'b-')
plt.plot(z, 'rx')

#5
plt.figure(2)      
plt.xlabel('k')
plt.ylabel('P(k)')
plt.axis([0, N, 0, 1.0])
plt.plot(P, 'b-')
plt.show()
