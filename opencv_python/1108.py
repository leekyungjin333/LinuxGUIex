#1108.py
''' ref1: https://github.com/leestott/IrisData
    ref2: https://en.wikipedia.org/wiki/Iris_flower_data_set
'''

import cv2
import numpy as np

#1
# Petal Length , Petal Width , Sepal Length , Sepal width, and Class
def loadIrisData(fileName):
  x_list = []
  y_list = []
  f = open(fileName, 'r')
  for line in f:
    line = line.rstrip('\n')             
    sVals = line.split(',')              
    fVals = list(map(np.float32, sVals))
    x_list.append(fVals[:-3]) # features
    y_list.append(fVals[-3:]) # class, one-hot-encoding  
  f.close()
  x = np.array(x_list, dtype=np.float32) # features
  y = np.array(y_list, dtype=np.float32) # class, one-hot-encoding
  return x, y  

x_train, y_train=loadIrisData('./data/irisTrainData.txt')
x_test,  y_test =loadIrisData('./data/irisTestData.txt')

#2
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([4, 5, 3]))
##ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,
                       1000,1e-5))


##ret=ann.train(samples=x_train,
##              layout=cv2.ml.ROW_SAMPLE,
##              responses=y_train)

trainData = cv2.ml.TrainData_create(samples=x_train,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y_train)
ret = ann.train(trainData)

#3
y_target = np.argmax(y_train, axis=1)
ret, res = ann.predict(x_train)
y_predict = np.argmax(res, axis = 1)
accuracy = np.sum(y_target==y_predict)/len(y_target)
print('x_train: accuracy=', accuracy)

#4
y_target = np.argmax(y_test, axis=1)
ret, res = ann.predict(x_test)
y_predict = np.argmax(res, axis = 1)
accuracy = np.sum(y_target==y_predict)/len(y_target)
print('x_test: accuracy=', accuracy)
