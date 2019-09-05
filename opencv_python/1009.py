# 1009.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''ref: Greg Welch and Gary Bishop, 'An Introduction to the Kalman Filter', 2006.
        Estimating a Random Constant : on-line with cv2.KalmanFilter'''
#1
x = -0.37727 # the truth value

q = 1e-5    #  process noise covariance
r = 0.01    #  measurement noise covariance, 1, 0.0001

KF = cv2.KalmanFilter(1, 1, 0)  # B = 0
KF.transitionMatrix    = np.ones((1, 1))  # A = 1
KF.measurementMatrix   = np.ones((1, 1))  # H = 1
KF.processNoiseCov     = q * np.eye(1)    # Q
KF.measurementNoiseCov = r * np.eye(1)    # R

#2 initial value
KF.errorCovPost        = np.ones((1, 1))  # P0 = 1
KF.statePost           = np.zeros((1, 1)) # x0 = 0

N = 50
X = [KF.statePost[0,0]]        # initial value
P = [KF.errorCovPost[0,0]]     # initial errorCovPost

#3
fig = plt.figure()
fig.canvas.set_window_title('Kalman Filter')
ax = plt.axes(xlim=(0, N), ylim=(x-3*np.sqrt(r), x+3*np.sqrt(r)))
ax.grid()
line1, = ax.plot([], [], 'b-', lw=2)
line2, = ax.plot([], [], 'rx')
line3, = ax.plot([0, N], [x, x], 'g-')  # the truth value line
xrange = np.arange(N)
Z = [] # for displaying measurements
#4
def init():
    for k in range(N):
        predict = KF.predict()
        z = np.random.randn(1, 1)*np.sqrt(r) + x  # measurement
        estimate = KF.correct(z[0])
        X.append(estimate[0,0])
        Z.append(z[0][0])
    line1.set_data(xrange, X)
    line2.set_data(xrange, Z)
##    line2.set_data([N-1], z)     
    return line1,line2

#5
def animate(k):
    global X, Z

    predict = KF.predict()
    z = np.random.randn(1, 1)*np.sqrt(r) + x  # measurement
    estimate = KF.correct(z[0])

    X = X[1:N]
    X.append(estimate[0,0])

    Z = Z[1:N]
    Z.append(z[0][0])
    line1.set_data(xrange, X)
    line2.set_data(xrange, Z)
##    line2.set_data([N-1], z)
    return line1,line2
#6
ani=animation.FuncAnimation(fig, animate,init_func=init,interval=25,blit=True)
plt.show()
