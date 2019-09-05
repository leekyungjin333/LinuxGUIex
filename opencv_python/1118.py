#1118.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#1
mnist = input_data.read_data_sets("./dnn/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
##x_train = x_train.reshape(-1, 28, 28, 1)
##x_test  = x_test.reshape(-1, 28, 28, 1)
print("x_train.shape=", x_train.shape)
print("y_train.shape=", y_train.shape)


#2, Define Tensorflow model
K = 32       # convolutional layer 1 output
L = 64       # convolutional layer 2 output
M = 128      # convolutional layer 3 output
N = 1024     # fully connected layer output
n_class = 10 # output layer

##tf.set_random_seed(0)
def CNN(x):
# weights
    W1 = tf.Variable(tf.random_normal([3, 3, 1, K], stddev=0.1))# 3x3 patch, 1 input channel, K output channel
    W2 = tf.Variable(tf.random_normal([3, 3, K, L], stddev=0.1))# 3x3xK conv, L outputs
    W3 = tf.Variable(tf.random_normal([3, 3, L, M], stddev=0.1))# 3x3xL conv, M outputs
    W4 = tf.Variable(tf.random_normal([M*4*4, N],   stddev=0.1))# FC Mx4x4 inputs, N outputs
    W5 = tf.Variable(tf.random_normal([N, n_class], stddev=0.1))# FC N inputs, 10 outputs
# bias
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [n_class]))

    Y1 = tf.nn.relu(tf.nn.conv2d(x, W1,
                    strides=[1,1,1,1],padding='SAME')+ B1)
    print('Y1.shape=', Y1.shape) # Y1.shape= (?, 28, 28, 32)
                   
    P1 = tf.nn.max_pool(Y1, ksize=[1,2,2,1], 
                          strides=[1,2,2,1], padding='SAME')
    print('P1.shape=', P1.shape) # P1.shape= (?, 14, 14, 32)
                 
    Y2 = tf.nn.relu(tf.nn.conv2d(P1, W2,
                    strides=[1,1,1,1], padding='SAME')+ B2)
    print('Y2.shape=', Y2.shape) # Y2.shape= (?, 14, 14, 64)
    
    P2 = tf.nn.max_pool(Y2, ksize=[1,2,2,1], 
                          strides=[1,2,2,1], padding='SAME')
    print('P2.shape=', P2.shape) # P2.shape= (?, 7, 7, 64)
    
    Y3 = tf.nn.relu(tf.nn.conv2d(P2, W3,
                    strides=[1,1,1,1], padding='SAME')+ B3)
    print('Y3.shape=', Y3.shape) # Y3.shape= (?, 7, 7, 128)
        
    P3 = tf.nn.max_pool(Y3, ksize=[1,2,2,1], 
                          strides=[1,2,2,1], padding='SAME')
    print('P3.shape=', P3.shape) # P3.shape= (?, 4, 4, 128)
                 
    _P3 = tf.reshape(P3, shape=[-1, M*4*4])
    print('_P3.shape=', _P3.shape)   # _P3.shape= (?, 2048)
    
    Y4 = tf.nn.relu(tf.matmul(_P3, W4) + B4)
    print('Y4.shape=', Y4.shape) # Y4.shape = (?, 1024)

    output = tf.matmul(Y4, W5) + B5
    print('output.shape=', output.shape) # output.shape= (?, 10)
    return output

#3
X  = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32,  [None, n_class])
Ylogits   = CNN(X)
y_predict = tf.nn.softmax(Ylogits, name='softmax')
print('X.shape=', X.shape) # X.shape= (?, 28, 28, 1)
print('Y.shape=', Y.shape) # Y.shape= (?, 10)
print('Ylogits.shape=', Ylogits.shape) # Ylogits.shape= (?, 10)
print('y_predict.shape=', y_predict.shape)#y_predict.shape= (?, 10)

##cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y)
loss = tf.reduce_mean(cross_entropy)

##optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
optimizer  = tf.train.AdamOptimizer(0.001).minimize(loss)

batch_size = 400
iter_per_epoch = mnist.train.num_examples//batch_size
print('iter_per_epoch=', iter_per_epoch)

#4
train_loss_list     = []
saver = tf.train.Saver() 
print('session start...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
               
    for epoch in range(100):       
        avg_loss = 0
        for i in range(iter_per_epoch): # mini batch           
            x_batch, y_batch = mnist.train.next_batch(batch_size)          
            x_batch = x_batch.reshape(-1, 28, 28, 1) 
            sess.run(optimizer,feed_dict={X: x_batch, Y: y_batch })
            loss_batch = sess.run(loss, feed_dict={X:x_batch, Y:y_batch})
            avg_loss += loss_batch
        avg_loss /= iter_per_epoch
        train_loss_list.append(avg_loss)
        print('epoch={}, loss={}'.format(epoch, avg_loss))
    print('train end..')

    # Save model: checkpoint, graph.pb
    saver.save(sess, 'dnn/MINIST_CNN_model2.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), 'dnn/', 'MINIST_CNN_graph2.pb')
    
    correct = tf.equal(tf.argmax(y_predict, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    x_test  = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    print('test accuracy :%.2f'%sess.run(accuracy*100,feed_dict={X:x_test, Y:y_test}))
    print('train accuracy:%.2f'%sess.run(accuracy*100,feed_dict={X:x_train[:10000], Y:y_train[:10000]}))
       
#5
str_cmd = ('python freeze_graph.py --input_graph      ./dnn/MINIST_CNN_graph2.pb'
                                 ' --input_checkpoint ./dnn/MINIST_CNN_model2.ckpt'
                                 ' --output_graph     ./dnn/MINIST_CNN_frozen_graph2.pb'
                                 ' --output_node_names  softmax')
from subprocess import PIPE, run
result = run(str_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
print('result.returncode=', result.returncode)
print('result.stdout=', result.stdout)
print('result.stderr=', result.stderr)

#6
x = list(range(len(train_loss_list)))
plt.plot(x, train_loss_list, label='train_loss')
plt.legend(loc='best')
plt.show()
