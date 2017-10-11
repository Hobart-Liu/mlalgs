import numpy as np
from NN import *

x = np.matrix([[0,0],
               [0,1],
               [1,0],
               [1,1]])
x = x.T

y = np.matrix([[0,1],
               [1,0],
               [1,0],
               [0,0]])
y = y.T


bpnn1 = neural_network_with_one_hidden_layer(2,6,2,learning_rate=0.8)
for i in range(5000):
    n = np.random.randint(0,4)
    tr_x = x[:,n]
    tr_y = y[:,n]
    error = bpnn1.train_one_iteration(tr_x, tr_y)
    bpnn1.reset_deltas()
    if i % 100 == 0:
        print(error)

print("======== Summary =======================")
cost, rate = bpnn1.test(x, y)
print("            Cost = ", cost)
print("Correctness rate = ", rate)

_,_,_, output = bpnn1.feedforward(x)
print("predict output:")
print(output)
print("target:")
print(y)