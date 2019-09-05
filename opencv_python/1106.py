#1106.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
N = 50
f1 = (0.6 + 0.6*np.random.rand(N), 0.5+0.6*np.random.rand(N))
f2 = (0.3 + 0.4*np.random.rand(N), 0.4+0.3*np.random.rand(N))
f3 = (0.8 + 0.4*np.random.rand(N), 0.3+0.3*np.random.rand(N))
f4 = (0.2*np.random.rand(N),       0.3*np.random.rand(N))

y_1ofN = np.zeros((4*N, 4), dtype = np.float32) # one-hot-encoding
##X = np.zeros((4*N, 2), dtype = np.float32)    # (x, y) 
##for i in range(N):
##  x = f1[0][i]
##  y = f1[1][i]
##  X[i] = [x,y]
##  y_1ofN[i] = [1,0,0,0]
##  
##  x = f2[0][i]
##  y = f2[1][i]
##  X[N+i] = [x,y]
##  y_1ofN[N+i] = [0,1,0,0]
##    
##  x = f3[0][i]
##  y = f3[1][i]
##  X[2*N+i] = [x,y]
##  y_1ofN[2*N+i] = [0,0,1,0]
##
##  x = f4[0][i]
##  y = f4[1][i]
##  X[3*N+i] = [x,y]
##  y_1ofN[3*N+i] = [0,0,0,1]
  
x = np.hstack((f1[0], f2[0], f3[0],f4[0])).astype(np.float32)
y = np.hstack((f1[1], f2[1], f3[1],f4[1])).astype(np.float32)
X = np.vstack((x, y)).T

y_1ofN[:N,:]     = [1,0,0,0] # one-hot-encoding 
y_1ofN[N:2*N,:]  = [0,1,0,0]
y_1ofN[2*N:3*N,:]= [0,0,1,0]
y_1ofN[3*N:,:]   = [0,0,0,1]

#2
ann = cv2.ml.ANN_MLP_create()
##ann.setLayerSizes(np.array([2, 4]))
##ann.setLayerSizes(np.array([2,5, 4]))
##ann.setLayerSizes(np.array([2, 5, 5, 4]))
ann.setLayerSizes(np.array([2, 5, 5, 5, 4]))

##ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                        cv2.TERM_CRITERIA_COUNT,1000,1e-5))
##ret=ann.train(samples=X,layout=cv2.ml.ROW_SAMPLE,responses=y_1ofN)
trainData = cv2.ml.TrainData_create(samples=X,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y_1ofN)
ret = ann.train(trainData)

#3
y_target = np.argmax(y_1ofN, axis=1)

ret, res = ann.predict(X)
y_predict = np.argmax(res, axis = 1)
accuracy = np.sum(y_target==y_predict)/len(y_target)
print('accuracy=', accuracy)

#4
h = 0.01
x_min, x_max = X[:, 0].min()-h, X[:, 0].max()+h
y_min, y_max = X[:, 1].min()-h, X[:, 1].max()+h

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

sample = np.c_[xx.ravel(), yy.ravel()]
ret, res = ann.predict(sample) 
Z = np.argmax(res, axis=1)
Z = Z.reshape(xx.shape)

markers= ('o','x','s','+','*','d')
colors = ('b','g','c','m','y','k')
labels = ('f1', 'f2', 'f3', 'f4')

fig = plt.gcf()
fig.set_size_inches(6,6)
##plt.contourf(xx, yy, Z, cmap=plt.cm.gray)
plt.contour(xx, yy, Z, colors='red', linewidths=2)

for i, k in enumerate(np.unique(y_target)):
  plt.scatter(X[y_target == k, 0],X[y_target==k, 1],
              c=colors[i], marker=markers[i], label=labels[k])
plt.legend(loc='best')

plt.show()
