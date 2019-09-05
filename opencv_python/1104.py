#1104.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
mlp_net = cv2.ml.ANN_MLP_create()
ann = mlp_net.load('./data/ann-xor.train')

#2
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype = np.float32)

y = np.array([0,1, 1,0], dtype = np.float32)   # XOR
target = y.copy()

#3
h = 0.01
x_min, x_max = X[:, 0].min()-h, X[:, 0].max()+h
y_min, y_max = X[:, 1].min()-h, X[:, 1].max()+h

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

sample = np.c_[xx.ravel(), yy.ravel()]
ret, Z = ann.predict(sample) 
Z = np.round(Z)
Z = Z.reshape(xx.shape)

fig = plt.gcf()
fig.set_size_inches(5,5)

##plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.contourf(xx, yy, Z, cmap=plt.cm.gray)
plt.contour(xx, yy, Z, colors='red', linewidths=3)
plt.scatter(*X[:, :].T, c=target, s = 75)
plt.show()
