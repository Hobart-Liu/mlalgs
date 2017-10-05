"""Percentron python implementation"""

import numpy as np

"""

参见李航的统计学习方法P34
注： x_j * x_i 在这里等于 G[j, i]

"""



class BinaryPercentron_Dual(object):
    def __init__(self):
        pass

    def fit(self, x, y, lr = 0.1, max_iter=500, disp=False):
        """
        李航的统计学习方法， 第二章，感知机的原始形式

        :param x: numpy array (number, vectors)
        :param y: numpy array (number, 1)
        :param lr: learning rate
        :param mat_iter: max iteration
        :return: w, b
        """

        assert(x.ndim == 2)
        assert(y.ndim == 2)

        m, n = x.shape

        assert(m > 1)

        w = np.zeros(shape=(m), dtype=np.float32)
        b = 0.0

        G = np.dot(x, x.T)
        if disp: print(G)

        for iter in range(max_iter):
            if disp: print(str(iter).center(50, '-'))
            done = True
            for i in range(m):


                # 按算法2.2的实现，用了循环，比较没效率
                # f = 0
                # for j in range(m):
                #     if disp: print("i = ", i, "j = ", j)
                #     f = f + sum(w[j] * y[j] * G[j, i])
                # f = y[i] * (f + b)

                # 简化为矩阵运算
                f = y[i] * (sum(w * y.T[0] * G[:, i])+b)

                if f <= 0:
                    w[i] = w[i] + lr
                    b = b + lr * y[i]
                    if disp: print("w = ", w, "b = ", b)
                    done = False
            if done:
                w = np.dot(y.T[0] * w, x)
                return w, b

        return None, None

    def predict(self, x, w, b):
        return [1 if sum(w * xx) + b > 0 else - 1 for xx in x]




perc = BinaryPercentron_Dual()


# 统计学习方法的例子
x = np.array([[3,3],[4,3],[1,1]])
y = np.array([1, 1, -1]).reshape(-1, 1)
w, b = perc.fit(x, y, 1, max_iter=10)
print("w:", w)
print("b:", b)
pred = perc.predict(x, w, b)
for i in range(len(y)):
    print("Expected value: %+d, predicted value: %+d" %(y[i], pred[i]))


# Example from https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
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

y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]).reshape(-1, 1)

w, b = perc.fit(x, y, 0.1)
print("w:", w)
print("b:", b)
pred = perc.predict(x, w, b)
for i in range(len(y)):
    print("Expected value: %+d, predicted value: %+d" %(y[i], pred[i]))

"""
w: [ 1.  1.]
b: [-3.]
Expected value: +1, predicted value: +1
Expected value: +1, predicted value: +1
Expected value: -1, predicted value: -1
w: [ 0.2065364  -0.23418118]
b: [-0.1]
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


