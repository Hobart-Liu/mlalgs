import tensorflow as tf
from os.path import join, expanduser
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""
one hidden layer
loss function = cross entropy 

"""


# Read data
datapath = join(expanduser("~"), "mldata/mnist")
mnist = input_data.read_data_sets(datapath, one_hot=True)
print("Train size:", mnist.train.images.shape, mnist.train.labels.shape)
print("Valid size:", mnist.validation.images.shape, mnist.validation.labels.shape)
print("Test  size:", mnist.test.images.shape, mnist.test.labels.shape)

# basic functions
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

input_dim = 784
hid1_dim = 128
output_dim = 10

batch_size = 100
num_steps = 1001

graph = tf.Graph()

with graph.as_default():

    # Variables
    x = tf.placeholder(tf.float32, shape=[None, input_dim])
    y = tf.placeholder(tf.float32, shape=[None, output_dim])
    W1 = tf.Variable(tf.truncated_normal(shape=[input_dim, hid1_dim], mean=0, stddev=0.1, dtype=tf.float32))
    W2 = tf.Variable(tf.truncated_normal(shape=[hid1_dim, output_dim], mean=0, stddev=0.1, dtype=tf.float32))
    b1 = tf.Variable(tf.zeros(shape=[hid1_dim], dtype=tf.float32))
    b2 = tf.Variable(tf.zeros(shape=[output_dim], dtype=tf.float32))
    tf_valid = tf.Variable(mnist.validation.images)
    tf_test = tf.Variable(mnist.test.images)

    global_step = tf.Variable(0)


    # Computation
    h1 = tf.nn.relu(tf.nn.xw_plus_b(x, W1, b1))
    logits = tf.nn.xw_plus_b(h1, W2, b2)
    pred = tf.nn.softmax(logits)

    # loss and optimizer
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )

    learning_rate = tf.train.exponential_decay(1.0, global_step, 100, 0.9, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for step in range(num_steps):
        batch = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch[0], y:batch[1]}
        _, l, train_pred = session.run([optimizer, loss, pred], feed_dict=feed_dict)


        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(
                train_pred, batch[1]
            ))

            feed_dict = {x: mnist.validation.images}
            valid_pred = session.run(pred, feed_dict=feed_dict)


            print('Validation accuracy: %.1f%%' % accuracy(
                valid_pred, mnist.validation.labels
            ))

    feed_dict = {x: mnist.test.images}
    test_pred = np.array(session.run(pred, feed_dict=feed_dict))
    print('Test accuracy: %.1f%%' % accuracy(
        test_pred, mnist.test.labels
    ))







