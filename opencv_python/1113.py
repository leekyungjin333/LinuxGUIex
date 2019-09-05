#1113.py
''' ref1: http://yann.lecun.com/exdb/mnist/
    ref2: https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390
'''
import gzip
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE  = 28
PIXEL_DEPTH = 255
NUM_LABELS  = 10

#1
def extract_data(filename, num_images):
  '''Extract the images into a 4D tensor [image index, y, x, channels].
     Values are rescaled from [0, 255] down to [0, 1].
  '''
##  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
##    data = data/PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
    return data

def extract_labels(filename, num_images):
  '''Extract the labels into a vector of int64 label IDs.'''
##  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
  return labels

def ont_hot_encoding(y): # assume that y is 1-D array
    t = np.zeros((y.size, 10), dtype=np.float32)
    for i, row in enumerate(t):
        row[y[i]] = 1      
    return t
  
# Extract it into np arrays.
def load_MINIST(flatten=True, one_hot=True):
  x_train=extract_data('./data/train-images-idx3-ubyte.gz',60000)
  y_train=extract_labels('./data/train-labels-idx1-ubyte.gz',60000)
  x_test=extract_data('./data/t10k-images-idx3-ubyte.gz',10000)
  y_test=extract_labels('./data/t10k-labels-idx1-ubyte.gz',10000)

  if flatten:
    x_train= x_train.reshape(-1, IMAGE_SIZE*IMAGE_SIZE) # (60000, 784)
    x_test = x_test.reshape(-1, IMAGE_SIZE*IMAGE_SIZE)  # (10000, 784)
  if one_hot:
    y_train = ont_hot_encoding(y_train)
    y_test = ont_hot_encoding(y_test)    
  return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_MINIST()

#2
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([784, 100, 10]))
##ann.setLayerSizes(np.array([784, 50, 50, 10]))

##ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,20,1e-5))

trainData = cv2.ml.TrainData_create(samples=x_train,
                                    layout=cv2.ml.ROW_SAMPLE,
                                    responses=y_train)
ret = ann.train(trainData)

#3
train_loss_list     = []
train_accuracy_list = []
test_loss_list      = []
test_accuracy_list  = []

train_size = 60000
iters_num  = 100

for i in range(iters_num):
  ret = ann.train(trainData, flags=cv2.ml.ANN_MLP_UPDATE_WEIGHTS)

  y_target = np.argmax(y_train, axis=1)
  ret, res_train = ann.predict(x_train)
  y_predict = np.argmax(res_train, axis = 1)
  train_accuracy = np.sum(y_target==y_predict)/len(y_target)
  train_loss = np.sum((y_train-res_train)**2)
  train_loss /= x_train.shape[0] # 60000
  train_accuracy_list.append(train_accuracy)
  train_loss_list.append(train_loss)

  y_target = np.argmax(y_test, axis=1)
  ret, res_test = ann.predict(x_test)
  y_predict = np.argmax(res_test, axis = 1)
  test_accuracy = np.sum(y_target==y_predict)/len(y_target)
  test_loss = np.sum((y_test-res_test)**2)
  test_loss /= x_test.shape[0] # 10000
  test_accuracy_list.append(test_accuracy)
  test_loss_list.append(test_loss)
    
  print('train_accuracy[{}]={}, '.format(i, train_accuracy), end='')
  print('test_accuracy={}'.format(test_accuracy))
  
print('train_loss={}, '.format(train_loss), end='')
print('test_loss={}'.format(test_loss))
  
#4
ann.save('./data/ann-minist_2layer_100RPROP.train')
##ann.save('./data/ann-minist_3layer_50RPROP.train')

x = list(range(len(train_loss_list)))
plt.plot(x, train_loss_list, label='train_loss')
plt.plot(x, test_loss_list, label='test_loss')

plt.legend(loc='best')
plt.show()

plt.plot(x, train_accuracy_list, label='train_accuracy')
plt.plot(x, test_accuracy_list, label='test_accuracy')
plt.legend(loc='best')
plt.show()
