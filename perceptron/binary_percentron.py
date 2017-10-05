"""Percentron python implementation"""

import numpy as np

"""

Note: fit and fit2 are two implementations, 
fit tries to use matrix, but probably it is overdo, because I only need to pick the first negative number
fit2 tries to loop data one by one, which might be better (to save computatoin time)

"""



class BinaryPercentron(object):
    def __init__(self):
        pass

    def fit(self, x, y, lr = 0.1):
        """
        李航的统计学习方法， 第二章，感知机的原始形式

        对loss按w求导，更新w
        对loss按b求导，更新b

        w = w + lr * y[i] * x[i]
        b = b + lr * y[i]

        :param: x [m*n] array, each vector has n dimensions
        :param: y [1*m] array, +/- 1
        :return: loss (scalar), weight and bias

        """

        assert(x.ndim==2)
        assert(y.ndim==1)
        m, n = x.shape

        w = np.zeros(shape=(n, 1), dtype=np.float32)
        b = 0.0

        f = y.reshape(-1, 1) * (np.dot(x, w) + b)

        while((f<=0).any()):

            first_negative = np.where(f<=0)
            i = first_negative[0][0]
            w += lr * y[i] * x[i].reshape(-1, 1)
            b += lr * y[i]
            f = y.reshape(-1, 1) * (np.dot(x, w) + b)

        return w, b, f
    
    def fit2(self, x, y, lr=0.1, max_iter = 5000):
        """
        李航的统计学习方法， 第二章，感知机的原始形式
        训练数据集 T={(x1, y1), (x2, y2), ... (xn, yn)}
        y = {-1, +1}

        :param x:  list of [x1, x2, ..., xn]
        :param y:  list of +/- 1
        :param lr: learning rate, default = 0.1
        :param max_iter: maxium iteration, to prevent worst case
        :return: w, b
        """

        ar_x = np.array(x, dtype=np.float32)
        ar_y = np.array(y, dtype=np.float32)

        assert(ar_x.ndim == 2)
        assert(ar_x.shape[0] == ar_y.shape[0])

        m, n = ar_x.shape

        w = np.zeros(n, dtype=np.float32)
        b = np.array(0.0)


        for iter in range(max_iter):
            done = True
            for i in range(m):
                if ar_y[i] * (np.dot(w, ar_x[i]) + b) <= 0:
                    w = w + lr * ar_y[i] * ar_x[i]
                    b = b + lr * ar_y[i]
                    done = False
                    break

            if done: return w, b
        return None, None

    def predict(self, x, w, b):
        y = np.dot(x, w) + b
        return [-1 if i <= 0 else 1 for i in y]




perc = BinaryPercentron()


# 统计学习方法的例子
x = np.array([[3,3],[4,3],[1,1]])
y = np.array([1, 1, -1])
w, b = perc.fit2(x, y, 1)
print("w:", w)
print("b:", b)
pred = perc.predict(x, w, b)
for i in range(len(y)):
    print("Expected value: %+d, predicted value: %+d" %(y[i], pred[i]))


# Example from https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
x = [[2.7810836, 2.550537003],
     [1.465489372, 2.362125076],
     [3.396561688, 4.400293529],
     [1.38807019, 1.850220317],
     [3.06407232, 3.005305973],
     [7.627531214, 2.759262235],
     [5.332441248, 2.088626775],
     [6.922596716, 1.77106367],
     [8.675418651, -0.242068655],
     [7.673756466, 3.508563011]]

y = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

w, b = perc.fit2(x, y, 0.1)
print("w:", w)
print("b:", b)
pred = perc.predict(x, w, b)
for i in range(len(y)):
    print("Expected value: %+d, predicted value: %+d" %(y[i], pred[i]))



exit()



x = np.array([[3,3],[4,3],[1,1]])
y = np.array([1, 1, -1])
w, b, info = perc.fit(x, y, 1)
print("info:", info)
print("w:", w)
print("b:", b)
pred = perc.predict(x, w, b)
for i in range(len(y)):
    print("Expected value: %+d, predicted value: %+d" %(y[i], pred[i]))


"""
info: [[ 3.]
 [ 4.]
 [ 1.]]
w: [[ 1.]
 [ 1.]]
b: -3.0
Expected value: +1, predicted value: +1
Expected value: +1, predicted value: +1
Expected value: -1, predicted value: -1
"""


x = np.array([[2.7810836, 2.550537003],
              [1.465489372, 2.362125076],
              [3.396561688, 4.400293529],
              [1.38807019, 1.850220317],
              [3.06407232, 3.005305973],
              [7.627531214, 2.759262235],
              [5.332441248, 2.088626775],
              [6.922596716, 1.77106367],
              [8.675418651, -0.242068655],
              [7.673756466, 3.508563011]])

y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

w, b, info = perc.fit(x, y, 1)
print("info:", info)
print("w:", w)
print("b:", b)
pred = perc.predict(x, w, b)
for i in range(len(y)):
    print("Expected value: %+d, predicted value: %+d" %(y[i], pred[i]))


"""
info: [[  1.22892827]
 [  3.50488378]
 [  4.28952389]
 [  2.46599791]
 [  1.70943697]
 [  8.29195437]
 [  5.12226054]
 [  9.15018327]
 [ 17.48477562]
 [  6.63270486]]
w: [[ 2.06536388]
 [-2.3418119 ]]
b: -1.0
Expected value: -1, predicted value: -1
Expected value: -1, predicted value: -1
Expected value: -1, predicted value: -1
Expected value: -1, predicted value: -1
Expected value: -1, predicted value: -1
Expected value: +1, predicted value: +1
Expected value: +1, predicted value: +1
Expected value: +1, predicted value: +1
Expected value: +1, predicted value: +1
Expected value: +1, predicted value: +1

"""

