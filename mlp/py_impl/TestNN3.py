import pandas as pd
import numpy as np
from NN import *
import pickle


df = pd.read_csv('train.csv','r',delimiter=',')

test = np.matrix(df[0:9999])
valid = np.matrix(df[10000:12999])
train = np.matrix(df[13000:])

n_test_records  = len(test)
n_valid_records = len(valid)
n_train_records = len(train)

n_shape = 28*28
n_label = 10


WEIGHT_DUMP = 'weight_2.dump'

def get_next_batch(data, num = 100, batch=0):
    # pick lable information
    tmp = data[batch*num:(batch+1)*num, 0]
    lbl = np.matrix(np.zeros([num,n_label]))
    for r in range(num):
        lbl[r,tmp[r,0]]=1
    # pick image information
    img  = data[batch*num:(batch+1)*num, 1:]
    #increase batch
    if ((batch+2) * num - 1) > n_train_records:
        batch = 0
    else:
        batch += 1

    '''
    Note: lbl (one hot) and img format

    ----y1----    ----x1----
    ----y2----    ----x2----
    ----y3----    ----x3----
    [num * 10]    [num * 784]

    '''
    return lbl, img, batch

def txt(img, threshold=200):
    render = ''
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            if img[i,j] > threshold:
                render += '@@'
            else:
                render += '  '
        render += '\n'
    return render


try:
    f = open(WEIGHT_DUMP, 'rb')
    d = pickle.load(f)
    w1, w2, w3, b1, b2, b3 = d.get('w1'), d.get('w2'), d.get('w3'), d.get('b1'), d.get('b2'), d.get('b3')
    f.close()
except FileNotFoundError:
    w1, w2, w3, b1, b2, b3 = None, None, None, None, None, None



bpnn1 = neural_network_with_two_hidden_layer(n_shape,100,49,n_label,learning_rate = 0.02, w1=w1, w2=w2, w3=w3, b1=b1, b2=b2, b3=b3)

valid_lbl, valid_img, _ = get_next_batch(valid, num=n_valid_records, batch=0)
valid_x = valid_img.T
valid_y = valid_lbl.T

batch = 0
for i in range(10000):
    lbl, img, batch = get_next_batch(train, num=100, batch=batch)
    tr_y = lbl.T
    tr_x = img.T

    train_error = bpnn1.train_one_iteration(tr_x, tr_y)
    bpnn1.reset_deltas()
    if i % 500 == 0:
        valid_error, rate = bpnn1.test(valid_x, valid_y)
        print("training error", train_error, "\t validation error", valid_error, "\t rate of correct", rate)

        w1, w2, w3, b1, b2, b3 = bpnn1.getWeights()
        d = dict(w1=w1, w2=w2, w3=w3, b1=b1, b2=b2, b3=b3)
        with open(WEIGHT_DUMP, 'wb') as f:
            pickle.dump(d, f)
            f.close()


lbl, img, _ = get_next_batch(test, num=n_test_records, batch=0)
test_y = lbl.T
test_x = img.T

print("======== Summary =======================")
cost, rate = bpnn1.test(test_x, test_y)
print("            Cost = ", cost)
print("Correctness rate = ", rate)


# visualize the checking

for i in range(10):
    rnd = np.random.randint(0, n_test_records)
    _,_,_,_, output = bpnn1.feedforward(test_x[:,rnd])
    print("predict output: ", output.argmax(0), " target: ", test_y[:,rnd].argmax(0))
    print(txt(test_x[:,rnd].T.reshape(28,28)))

