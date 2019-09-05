#1122.py
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
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(train_faces.reshape(-1, HEIGHT, WIDTH), train_labels)
 
#3: predict test_faces using recognizer
correct_count = 0
for i, face in enumerate(test_faces.reshape(-1, HEIGHT, WIDTH)):    
    predict_label, confidence = recognizer.predict(face)
    if test_labels[i]== predict_label:
        correct_count+= 1
    print('test_labels={}: predicted:{}, confidence={}'.format(
                     test_labels[i], predict_label,confidence))
accuracy = correct_count / float(len(test_faces))
print('accuracy=', accuracy)
