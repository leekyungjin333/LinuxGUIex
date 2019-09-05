#1115.py
import tensorflow as tf
import random
import numpy as np

#1
x_train = [[0,0], [0,1], [1,0], [1,1]] # train input
y_train = [[1,0], [0,1], [0,1], [1,0]] # one-hot encoding, target

learning_rate = 0.1
n_input  = 2  # 입력층 노드 개수
n_hidden = 2  # 은닉층 노드 개수
n_class  = 2  # 출력층 노드 개수

#2
X = tf.placeholder('float', [None, n_input])
Y = tf.placeholder('float', [None, n_class ]) # target label

# hidden layer
W1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
b1 = tf.Variable(tf.random_normal([n_hidden]))
hidden_layer= tf.nn.sigmoid(tf.matmul(X,W1) + b1)

# output layer
W2 = tf.Variable(tf.random_normal([n_hidden, n_class]))
b2 = tf.Variable(tf.random_normal([n_class]))

output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, W2)+b2)
##y_predict= tf.identity(output_layer, name='identity')
y_predict = tf.nn.softmax(output_layer, name='softmax')

#3 loss and optimizer
loss =  tf.reduce_mean(tf.square(Y-y_predict)) 
##loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y_predict), 1))  # cross_entropy

##optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#4 Train
saver = tf.train.Saver()

print("session start...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch  in range(10001):      
        sess.run(optimizer,feed_dict={X: x_train, Y: y_train })
        if epoch%1000 == 0:
            print(epoch , sess.run(loss, feed_dict={X:x_train, Y:y_train}))
                
    saver.save(sess, 'dnn/XOR_model.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), 'dnn/', 'XOR_graph.pb')

    # test trained model
    predict = tf.argmax(y_predict, 1)
    target  = tf.argmax(Y, 1)
    print('predict:', sess.run(predict, feed_dict={X: x_train}))
    print('target:', sess.run(target,   feed_dict={Y: y_train}))

    correct=tf.equal(tf.argmax(y_predict,1),tf.argmax(Y,1))#tf.equal(predict,target)     
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print('accuracy: %.2f'%sess.run(accuracy*100,feed_dict={X:x_train, Y:y_train}))
##    print("accuracy %s"%(100*accuracy.eval({X: x_train, Y: y_train}))) 
         
#5
str_cmd = ('python freeze_graph.py --input_graph ./dnn/XOR_graph.pb'
                                 ' --input_checkpoint ./dnn/XOR_model.ckpt'
                                 ' --output_graph     ./dnn/XOR_frozen_graph.pb'
                                 ' --output_node_names  softmax') # identity
from subprocess import PIPE, run
result = run(str_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
print('result.returncode=', result.returncode)
print('result.stdout=', result.stdout)
print('result.stderr=', result.stderr)
