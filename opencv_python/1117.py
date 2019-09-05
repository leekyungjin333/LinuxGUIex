#1117.py
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

#1
mnist = input_data.read_data_sets('dnn/', one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

#2
learning_rate = 0.01
n_input    = 784 # 입력층 노드 개수
n_hidden   = 100 # 은닉층 노드 개수
n_class    = 10  # 출력층 노드 개수
batch_size = 400 # 미니배치 크기 

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])# target label

initializer = tf.initializers.random_normal()

# hidden layer
W1 = tf.Variable(initializer([n_input, n_hidden]))
b1    = tf.Variable(initializer([n_hidden]))
hidden_layer= tf.nn.sigmoid(tf.matmul(X,W1) + b1)

# output layer
W2 = tf.Variable(initializer([n_hidden, n_class]))
b2 = tf.Variable(initializer([n_class]))
output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, W2)+b2)
output_layer = tf.matmul(hidden_layer, W2)+b2
y_predict = tf.nn.softmax(output_layer, name='softmax')

# loss and optimizer
##loss =  tf.reduce_mean(tf.square(Y-y_predict)) 
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y_predict), axis=1)) # cross_entropy
##optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#3
saver = tf.train.Saver() 
print('session start...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch  in range(10001):
        x_batch, y_batch = mnist.train.next_batch(batch_size)      
        sess.run(optimizer,feed_dict={X: x_batch, Y: y_batch })
        
        if epoch%1000 == 0:
            print(epoch , sess.run(loss, feed_dict={X:x_train, Y:y_train}))
            
    # Save model: checkpoint, graph.pb
    saver.save(sess, 'dnn/MINIST_MLP_model.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), 'dnn/', 'MINIST_MLP_graph.pb')

    # accuracy
    correct=tf.equal(tf.argmax(y_predict,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print('train accuracy:%.2f'%sess.run(accuracy*100,feed_dict={X:x_train, Y:y_train}))
    print('test accuracy :%.2f'%sess.run(accuracy*100,feed_dict={X:x_test, Y:y_test}))
#4
str_cmd = ('python freeze_graph.py --input_graph      ./dnn/MINIST_MLP_graph.pb'
                                 ' --input_checkpoint ./dnn/MINIST_MLP_model.ckpt'
                                 ' --output_graph     ./dnn/MINIST_MLP_frozen_graph.pb'
                                 ' --output_node_names  softmax')
from subprocess import PIPE, run
result = run(str_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
print('result.returncode=', result.returncode)
print('result.stdout=', result.stdout)
print('result.stderr=', result.stderr)
