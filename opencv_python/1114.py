#1114.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
net = cv2.dnn.readNetFromTensorflow('./dnn/XOR_frozen_graph.pb')

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype = np.float32)

y = np.array([[1, 0],
              [0, 1],
              [0, 1],           
              [1, 0]], dtype = np.float32) # XOR

#2  
for x in X:
    blob = cv2.dnn.blobFromImage(x)  # blob.shape =  (1,1,1,2)
##    blob = x.reshape((1, 1, 1, 2))
    net.setInput(blob)
       
    res = net.forward()
    print(x, res)
    
    predict = np.argmax(res)
    print('predict =', predict )

#3    
h = 0.01
x_min, x_max = X[:, 0].min()-h, X[:, 0].max()+h
y_min, y_max = X[:, 1].min()-h, X[:, 1].max()+h

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

sample = np.float32(np.c_[xx.ravel(), yy.ravel()])
sample = sample.reshape((-1, 1, 1,  2)) # sample.reshape((len(sample),1,1,2))
net.setInput(sample)
res = net.forward()

Z = np.argmax(res, axis=1)
Z = Z.reshape(xx.shape)

fig = plt.gcf()
fig.set_size_inches(5,5)

plt.contourf(xx, yy, Z, cmap=plt.cm.gray)
plt.contour(xx, yy, Z,  colors='red')
plt.scatter(*X[:, :].T, c=y[:,0], s = 75)
plt.show()
