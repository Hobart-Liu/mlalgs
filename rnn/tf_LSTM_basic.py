import tensorflow as tf
import string
import numpy as np
from os.path import join, expanduser
import os
import zipfile
import random


vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '

def check_file(filename):
    statinfo = os.stat(filename)
    assert (statinfo.st_size == 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        # 直接读成一个字符串，因为后面用到的就是串数据
        data = tf.compat.as_str(f.read(f.namelist()[0]))
    return data

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - ord('a') + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0

def id2char(dictid):
    if dictid > 0:
        return chr(dictid + ord('a') - 1)
    else:
        return ' '

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""

        # we will use one-hot-vector, so initialize all with zero is fine
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """

        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or probability distribution over the possible
    characters back into its (most likely character representation"""

    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""

    s = [''] * batches[0].shape[0]  # batch_size
    for b in batches:
        # s = [''.join(x) for x in zip(s, characters(b))]
        s = [x+y for x, y in zip(s, characters(b))]
    return s

def logprob(predictions, labels):
    """log-probability of the true labels in a predicted batch"""
    predictions[predictions < 1e-10] = 1e-10

    # np.multiply(list, list) is element wise product

    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized probabilities"""

    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s>= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """Trun a (column) prediction into 1-hot encoded samples"""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    """Generate a random column of probabiliities"""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/(np.sum(b, 1).reshape(-1, 1))



if __name__ == "__main__":
    # load raw data
    root = join(expanduser("~"), 'mldata')
    file = join(root, 'text8/text8.zip')
    text = read_data(file)
    print("data size is %d" % len(text))

    # define the first 1000 words as validation, take the rest as training
    valid_size = 1000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    print(train_size, train_text[:64])
    print(valid_size, valid_text[:64])


    print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
    print(id2char(1), id2char(26), id2char(0))

    batch_size = 64
    num_unrollings = 10


    train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    print(batches2string(train_batches.next()))
    print(batches2string(train_batches.next()))
    for _ in range(2):
        print(batches2string(valid_batches.next()))

    num_nodes = 64

    graph = tf.Graph()
    with graph.as_default():

        # Parameters:
        # Input gate: input, previous output, and bias
        ix = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        im = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
        # Forget gate: input, previous output, and bias
        fx = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        fm = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
        # Memory cell: input, previous output, and bias
        cx = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        cm = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output, and bias
        ox = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, num_nodes], mean=-0.1, stddev=0.1))
        om = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes], mean=-0.1, stddev=0.1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))

        # Variable saving state across unrollings
        """
        saved_output 是向上的产出，saved_state 是自己的状态记忆。
        """
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state  = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

        # Classifer weights and bias
        w = tf.Variable(tf.truncated_normal(shape=[num_nodes, vocabulary_size], mean=-0.1, stddev=0.1))
        b = tf.Variable(tf.zeros([vocabulary_size]))


        # define cell computation
        def lstm_cell(i, o, state):
            """
            Create LSTM cell,
            See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            """

            input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
            update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
            state = forget_gate * state + input_gate*tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state


        # input data
        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(
                tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size])
            )
        train_inputs = train_data[:num_unrollings]
        train_labels = train_data[1:]  # labels are inputs shifted by one time step

        # unrolled LSTM loop
        outputs = list()
        output  = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings
        with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
            # Classifer
            logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.concat(train_labels, 0), logits=logits
                )
            )


        # Optimizer
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate=10.0,
                                                   global_step=global_step,
                                                   decay_steps=5000,
                                                   decay_rate=0.1,
                                                   staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step
        )

        # predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        # sample_input 是一个1-hot编码后的字符  ==> 经过同样的LSTM CELL得到下一个预测的字符 sample_prediction
        sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        sample_output, sample_state = lstm_cell(
            sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))


    num_steps = 7001
    summary_frequency = 100

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        mean_loss = 0
        for step in range(num_steps):
            batches = train_batches.next()
            feed_dict = dict()
            for i in range(num_unrollings + 1):
                feed_dict[train_data[i]] = batches[i]
            _, l, predictions, lr = session.run(
                [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)

            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print(
                    'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0


                labels = np.concatenate(list(batches)[1:])


                print('Minibatch perplexity: %.2f' % float(
                    np.exp(logprob(predictions, labels))))


                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    for _ in range(5):
                        feed = sample(random_distribution())
                        sentence = characters(feed)[0]
                        reset_sample_state.run()
                        for _ in range(79):
                            prediction = sample_prediction.eval({sample_input: feed})

                            feed = sample(prediction)

                            sentence += characters(feed)[0]
                        print(sentence)
                    print('=' * 80)


                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_size):
                    b = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: b[0]})
                    valid_logprob = valid_logprob + logprob(predictions, b[1])

                print('Validation set perplexity: %.2f' % float(np.exp(
                    valid_logprob / valid_size)))


