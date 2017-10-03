import tensorflow as tf
from os.path import join, expanduser
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""
no hidden layer, direct output from 784 to 10 softmax
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


# in this simple, example, use tf.InteractiveSession()

sess = tf.InteractiveSession()

W = tf.Variable(tf.truncated_normal(shape=[784, 10], mean=-0.1, stddev=0.1))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, shape=[100, 784])
logits = tf.nn.xw_plus_b(x, W, b)
y_ = tf.placeholder(tf.float32, shape=[100, 10])

pred = tf.nn.softmax(logits)

tf_valid = tf.constant(mnist.validation.images)
tf_test  = tf.constant(mnist.test.images)
valid_pred = tf.nn.softmax(
    tf.nn.xw_plus_b(tf_valid, W, b)
)
test_pred = tf.nn.softmax(
    tf.nn.xw_plus_b(tf_test, W, b)
)


loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
)

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess.run(tf.global_variables_initializer())

for step in range(2001):
    batch = mnist.train.next_batch(100)
    feed_dict = {
        x:  batch[0],
        y_: batch[1]
    }
    _, l, predictions = sess.run([optimizer, loss, pred], feed_dict=feed_dict)

    if (step % 100 == 0):
        print('Loss at step %d: %f' % (step, l))
        print('Training accuracy: %.1f%%' % accuracy(
            predictions, batch[1]
        ))

        print('Validation accuracy: %.1f%%' % accuracy(
            valid_pred.eval(), mnist.validation.labels
        ))

print('Test accuracy: %.1f%%' % accuracy(
    test_pred.eval(), mnist.test.labels
))




