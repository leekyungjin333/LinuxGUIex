#1109.py
''' ref1: https://github.com/leestott/IrisData
    ref2: https://en.wikipedia.org/wiki/Iris_flower_data_set
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
ann.setTermCriteria((cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,10,1e-5))

train_loss_list = []
train_accuracy_list = []

test_loss_list = []
test_accuracy_list = []

trainData = cv2.ml.TrainData_create(samples=x_train,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y_train)
ret = ann.train(trainData) # initial training

#3
iters_num = 1000
for i in range(iters_num):
  ret = ann.train(trainData, flags=cv2.ml.ANN_MLP_UPDATE_WEIGHTS)

#3-1
  y_target = np.argmax(y_train, axis=1)
  ret, res = ann.predict(x_train)
  y_predict = np.argmax(res, axis = 1)
  train_accuracy = np.sum(y_target==y_predict)/len(y_target)
  train_loss = np.sum((y_train-res)**2)

  train_accuracy_list.append(train_accuracy)
  train_loss_list.append(train_loss)

#3-2
  y_target = np.argmax(y_test, axis=1)
  ret, res = ann.predict(x_test)
  y_predict = np.argmax(res, axis = 1)
  test_accuracy = np.sum(y_target==y_predict)/len(y_target)
  test_loss = np.sum((y_test-res)**2)

  test_accuracy_list.append(test_accuracy)
  test_loss_list.append(test_loss)
    
  print('train_accuracy[{}]={}, '.format(i, train_accuracy), end='')
  print('train_loss={}'.format(train_loss))

#4
x = np.linspace(0, iters_num, num=iters_num)
plt.plot(x, train_loss_list, label='train_loss')
plt.plot(x, test_loss_list, label='test_loss')

plt.legend(loc='best')
plt.show()

plt.plot(x, train_accuracy_list, label='train_accuracy')
plt.plot(x, test_accuracy_list, label='test_accuracy')
plt.legend(loc='best')
plt.show()
