import tensorflow as tf
import numpy as np
from os.path import join, expanduser
import os
import zipfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
import random
import collections

# CBOW = Continuous Bag-of-Words
# skipgram 是通过target对应附近的单词
# CBOW是通过附近的一组单词对应target

def check_file(filename):
    statinfo = os.stat(filename)
    assert(statinfo.st_size==31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, vocabulary_size = 50000):

    # UNK token is used to denote words that are not in the dictionary
    # [count] set of tuples (word, count) with most common 50000 words
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count  # replace -1 with unknown count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def generate_batch(batch_size, skip_window):
    global data_index

    span = 2 * skip_window + 1 # [skip_window, target, skip_window]

    # build return value
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # build buffer holder and fill in buffer
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # fille in data, use to_avoid

    for i in range(batch_size):
        target = skip_window

        idx = 0
        for j in range(span):
            if j == span//2: continue
            batch[i, idx] = buffer[j]
            idx += 1

        labels[i, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    assert batch.shape[0] == batch_size and batch.shape[1] == span -1

    return batch, labels


def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  plt.show()

if __name__ == '__main__':
    root = join(expanduser("~"), 'mldata')
    file = join(root, 'text8/text8.zip')
    vocabulary_size = 50000

    words = read_data(file)
    print('data size is %d' % len(words))
    print(words[:30])

    data, count, dictionary, reverse_dictionary = build_dataset(words=words, vocabulary_size=vocabulary_size)
    print(data[:6])
    print(count[:6])
    print(list(dictionary.items())[:6])
    print(list(reverse_dictionary.items())[:6])

    del words

    data_index = 0
    batch, labels = generate_batch(batch_size=8, skip_window=4)
    print(batch)
    print(labels)
    for i in range(8):
        print('    batch:', [reverse_dictionary[bi] for bi in batch[i]])
        print('    labels:', [reverse_dictionary[li] for li in labels[i]])


    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector
    skip_window = 1 # How many words to consider left and right
    num_sampled = 64  # number of negative reocrds sampled in softmax

    valid_size = 16 # Random set of words to evaluate similarity on
    # pick 8 samples from 100
    valid_examples = np.array(random.sample(range(100), valid_size//2))
    # pick 8 samples from 1000-1100
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1100), valid_size//2))



    num_steps = 100001

    graph = tf.Graph()

    with graph.as_default():

        # input Variable
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2*skip_window])
        train_labels  = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Variables
        # embedding by its definiation, is alwasy [vacabulary_size, embedding_size]
        embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size],minval=-1.0, maxval=1.0))
        # since we are going to use sampled_softmax_loss, it defines weight in this way
        # weights: A Tensor of shape [num_classes, dim]
        softmax_weights = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        # again, sampled_softmax_loss defines bias in this way
        # biases: A Tensor of shape [num_classes]
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Computation
        embeds = tf.nn.embedding_lookup(embeddings, train_dataset)

        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases = softmax_biases,
                                       inputs=tf.reduce_mean(embeds,1), labels=train_labels,
                                       num_sampled=num_sampled, num_classes=vocabulary_size )
        )


        # Optimizer
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # Validation
        norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims = True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")

        average_loss = 0

        for step in range(num_steps):
            batch, labels = generate_batch(batch_size, skip_window)
            feed_dict = {train_dataset: batch, train_labels:labels}
            _, l = session.run([optimizer, loss], feed_dict = feed_dict)

            average_loss += l

            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                print('Average loss at step %d: %f' % (step, average_loss))

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # numb of nearest neightbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

        final_embeddings = normalized_embeddings.eval()

    num_points = 400

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

    words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
    plot(two_d_embeddings, words)