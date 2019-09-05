#1102.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([2, 1]))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_EPS+
                        cv2.TERM_CRITERIA_COUNT,1000,1e-5))

#2
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype = np.float32)

y = np.array([0, 0, 0, 1], dtype = np.float32)          # AND
##y = np.array([0, 1, 1, 1], dtype = np.float32)        # OR
target = y.copy()

#3
##ret = ann.train(samples=X, layout=cv2.ml.ROW_SAMPLE, responses=y)
trainData = cv2.ml.TrainData_create(samples=X,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y)
ret = ann.train(trainData)
##ret = ann.train(trainData, flags = cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
##ret = ann.train(trainData, flags = cv2.ml.ANN_MLP_NO_OUTPUT_SCALE+
##                                       cv2.ml.ANN_MLP_NO_INPUT_SCALE)

#3-1
w0 = ann.getWeights(0)
print('w0.shape=', w0.shape)
print('w0=', w0)

#3-2
w1 = ann.getWeights(1)
print('w1.shape=', w1.shape)
print('w1=', w1)

#3-3
w2 = ann.getWeights(2)
print('w2.shape=', w2.shape)
print('w2=', w2)

#3-4
w3 = ann.getWeights(3)
print('w3.shape=', w3.shape)
print('w3=', w3)

#4
print('ann.predict(x)')
for x in X:    
    print(x, ann.predict(x.reshape(-1, 2)))

#5    
dst = np.zeros((512,512,3), np.uint8)
rows, cols = dst.shape[:2]

for y in range(rows):
    y1 = (rows - y)/rows  # upside-down, [0, 1]
    for x in range(cols):
        x1 = x/cols
        sample = np.array([x1, y1], dtype=np.float32).reshape(-1, 2)
        _, res = ann.predict(sample)
        if int(np.round(res[0])) == 0:
            dst[y, x] = (0, 0, 0)
        else:
            dst[y, x] = (200, 200, 200)
cv2.imshow('dst', dst)
##cv2.waitKey()
##cv2.destroyAllWindows() 

#6
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
