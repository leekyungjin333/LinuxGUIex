#1107.py
'''
   ref1: https://github.com/leestott/IrisData
   ref2: https://en.wikipedia.org/wiki/Iris_flower_data_set
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
# Petal Length , Petal Width , Sepal Length , Sepal width, and Class
def loadIrisData(fileName):
##  resultList = []
  x_list = []
  y_list = []
  f = open(fileName, 'r')
  for line in f:
    line = line.rstrip('\n')             
    sVals = line.split(',')              
    fVals = list(map(np.float32, sVals))
    x_list.append(fVals[:-3]) # features
    y_list.append(fVals[-3:]) # class, one-hot-encoding  
##    resultList.append(fVals)
  f.close()

  x = np.array(x_list, dtype=np.float32) # features
  y = np.array(y_list, dtype=np.float32) # class, one-hot-encoding
  
##  data = np.array(resultList, dtype=np.float32) # np.asarray
##  X =  data[:,:-3].copy() # features
##  y = data[:,-3:].copy()  # class, one-hot-encoding
  return x, y

x_train, y_train=loadIrisData('./data/irisTrainData.txt')
x_test,  y_test = loadIrisData('./data/irisTestData.txt')
print('x_train.shape=', x_train.shape)
print('y_train.shape=', y_train.shape)
print('x_test.shape=',  x_test.shape)
print('y_test.shape=',  y_test.shape)

#2
y_target = np.argmax(y_train, axis = 1)

markers= ('o','x','s','+','*','d')
colors = ('b','g','c','m','y','k')
labels = ['Iris setosa','Iris virginica','Iris versicolor']

fig = plt.gcf()
fig.set_size_inches(6,6)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
for i, k in enumerate(np.unique(y_target)):
  plt.scatter(x_train[y_target== k, 0], # Petal Length
              x_train[y_target== k, 1], # Petal Width
              c=colors[i], marker=markers[i], label=labels[i])
plt.legend(loc='best')
plt.show()

#3
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
for i, k in enumerate(np.unique(y_target)):
  plt.scatter(x_train[y_target== k, 2], # Sepal Length
              x_train[y_target== k, 3], # Sepal Width
              c=colors[i], marker=markers[i], label=labels[i])
plt.legend(loc='best')
plt.show()
