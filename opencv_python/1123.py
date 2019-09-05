#1123.py
import cv2
import numpy as np
import random

#1
WIDTH = 92
HEIGHT = 112
def load_face(filename='./data/faces.csv', test_ratio=0.2):
    file = open(filename, 'r')
    lines = file.readlines()

    N = len(lines)
    faces = np.empty((N, WIDTH*HEIGHT), dtype=np.uint8 )
    labels = np.empty(N, dtype = np.int32)
    for i, line in enumerate(lines):
        filename, label = line.strip().split(';')
        labels[i] = int(label)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        faces[i, :] = img.flatten()
  
# shuffling and seperate train and test data
    indices = list(range(N))
    random.seed(1) # same random sequences, so the same result
    random.shuffle(indices)
    shuffle_faces = faces[indices]
    shuffle_labels = labels[indices]

    test_size = int(test_ratio*N)

    test_faces = shuffle_faces[:test_size]
    test_labels = shuffle_labels[:test_size]

    train_faces = shuffle_faces[test_size:]
    train_labels = shuffle_labels[test_size:]
    return train_faces, train_labels, test_faces, test_labels

train_faces, train_labels, test_faces, test_labels = load_face()
print('train_faces.shape=',  train_faces.shape)
print('train_labels.shape=', train_labels.shape)
print('test_faces.shape=',   test_faces.shape)
print('test_labels.shape=',  test_labels.shape)

#2
N = 80
mean, eigenFace = cv2.PCACompute(train_faces, mean=None, maxComponents=N)
print('mean.shape = ', mean.shape)
print('eigenFace.shape=',  eigenFace.shape)

Y =cv2.PCAProject(train_faces, mean, eigenFace) # train result
print('Y.shape=', Y.shape)

#3: display eigen Face
dst = np.zeros((8*HEIGHT, 10*WIDTH), dtype=np.uint8)
for i in range(N):
  x = i%10
  y = i//10
  x1 = x*WIDTH
  y1 = y*HEIGHT
  x2 = x1+WIDTH
  y2 = y1+HEIGHT  
  
  img = eigenFace[i].reshape(HEIGHT, WIDTH)
  dst[y1:y2, x1:x2] = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
cv2.imshow('eigenFace %s'%N, dst)

#4: predict test_faces using  y = eigenFace(face - mean) and min distance
correct_count = 0
for i, face in enumerate(test_faces): 
    y =cv2.PCAProject(face.reshape(1,-1), mean, eigenFace)
    dist=np.sqrt(np.sum((Y-y)**2,axis=1))
    k = np.argmin(dist)
    predict_label = train_labels[k]

    if test_labels[i]== predict_label:
        correct_count+= 1
    print('test_labels={}: predicted:{}'.format(
                     test_labels[i], predict_label))
    
accuracy = correct_count / float(len(test_faces))
print('accuracy=', accuracy)
