import tensorflow as tf
from os.path import join, expanduser
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""
Le-Net5
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

img_size = 28
num_channels = 1 # grayscale
num_labels = 10
batch_size = 128

depth1 = 6
depth2 = 16
patch1         = [5, 5, num_channels, depth1]
patch2         = [5, 5, depth1, depth2]
pooling_kernel = [1, 2, 2, 1]  # [batch, width, high, channel]

fc_hid1 = 120
fc_hid2 = 84

graph = tf.Graph()
with graph.as_default():

    # Variable
    x = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, num_channels])
    y = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    tf_valid_dataset = tf.constant(mnist.validation.images.reshape(-1, img_size,img_size, 1))
    tf_test_dataset  = tf.constant(mnist.test.images.reshape(-1, img_size, img_size, 1))

    # conv layer
    W1 = tf.Variable(tf.truncated_normal(shape=[5, 5, num_channels, depth1], stddev=0.1))
    B1 = tf.Variable(tf.zeros(shape=[depth1]))
    W2 = tf.Variable(tf.truncated_normal(shape=[5, 5, depth1, depth2], stddev=0.1))
    B2 = tf.Variable(tf.constant(1.0, shape=[depth2], dtype=tf.float32))
    W3 = tf.Variable(tf.truncated_normal(shape=[img_size // 4 * img_size // 4 * depth2, fc_hid1], stddev = 0.1))
    B3 = tf.Variable(tf.constant(1.0, shape=[fc_hid1], dtype=tf.float32))
    W4 = tf.Variable(tf.truncated_normal(shape=[fc_hid1, fc_hid2], stddev=0.1))
    B4 = tf.Variable(tf.constant(1.0, shape=[fc_hid2], dtype=tf.float32))
    W5 = tf.Variable(tf.truncated_normal(shape=[fc_hid2, num_labels], stddev=0.1))
    B5 = tf.Variable(tf.constant(1.0, shape=[num_labels], dtype=tf.float32))
    global_step = tf.Variable(0)
    lr = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step,
                                    decay_steps=100, decay_rate=0.98, staircase=True)



    # computation
    def model(data):
        conv = tf.nn.conv2d(input=data, filter=W1, strides=[1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + B1)
        pool = tf.nn.max_pool(value=hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(input=pool, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + B2)
        pool = tf.nn.max_pool(value=hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.nn.xw_plus_b(reshape, W3, B3))
        dr = tf.nn.dropout(hidden, 0.5)
        hidden = tf.nn.relu(tf.nn.xw_plus_b(dr, W4, B4))
        return tf.nn.xw_plus_b(hidden, W5, B5)

    logits = model(x)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

    # predictions for the training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction  = tf.nn.softmax(model(tf_test_dataset))

num_steps = 5001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):

        batch = mnist.train.next_batch(batch_size)
        train_x = batch[0].reshape(batch_size, img_size, img_size, 1)
        train_y = batch[1]
        feed_dict = {x:train_x, y:train_y}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict = feed_dict
        )

        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, train_y))

            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), mnist.validation.labels))

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), mnist.test.labels))

