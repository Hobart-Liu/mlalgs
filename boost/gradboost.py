"""
This implementation refers to lin XuanTian Machine Learning @ coursera

Recipe:

s1 = s2 = ... = sn = 0
for t = 1, 2, ..., T
    obain g_t by A({(xn, yn - sn)}) where A is a (squared-error) regression algorithm
    (how about sampled and pruned CART)
    compute alpha_t = One VarLinearRegression({(g_t(xn), yn - sn)})
    update sn <- sn + alpha_t g_t(xn)
return G(X) = sum(a_t g_t(x))

"""

import numpy as np
import matplotlib.pyplot as plt


def lost(x, y):  #mse
    return np.sum([(y - x_) ** 2 for x_ in x])


class CARTStump(object):
    def __init__(self):
        self.split_point = None
        self.feat = None
        self.less_than_result = None  # <=
        self.larger_than_result = None  # >

    # def get_y_hat(self, data, weight):
    #     n = len(data)
    #     return np.dot(weight.reshape(1,-1), data.reshape(-1, 1))/n


    def fit(self, X, Y):
        """
        simple function to handle 2D training data
        :param X: 2D dim
        :param Y: 1D vector, label
        :return:
        """
        m, n = X.shape
        the_min = np.inf

        for feat in range(n):
            data = X[:, feat]
            for row in range(m - 1):
                split_point = X[row, feat]
                sub1 = Y[data <= split_point]
                sub2 = Y[data > split_point]
                y_hat_1 = sub1.mean()
                y_hat_2 = sub2.mean()
                mean_sequre_error = lost(sub1, y_hat_1) + lost(sub2, y_hat_2)
                # print("Point %5.2f, mse %5.2f" %(split_point, mean_sequre_error))
                if mean_sequre_error < the_min:
                    the_min = mean_sequre_error
                    self.split_point = split_point
                    self.feat = feat
                    self.less_than_result = y_hat_1
                    self.larger_than_result = y_hat_2
        # print("best point %5.2f, mse %5.2f" % (self.split_point, the_min))
        return the_min

    def predict(self, x):
        if x[self.feat] <= self.split_point:
            return self.less_than_result
        else:
            return self.larger_than_result


class GradientBoost:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.alphas = []
        self.stumps = []


    def fit(self, max_iter=5):
        iter = 0
        S = np.zeros(len(self.Y))
        G = np.zeros(len(self.Y))
        costs = []
        while iter < max_iter:
            stump = CARTStump()
            print("====> iter", iter)
            cost = stump.fit(self.X, self.Y - S)
            G = np.array([stump.predict(x) for x in self.X])
            a = np.sum([g * (y-s) for g, y, s in zip(G, self.Y, S)])/np.sum([g**2 for g in G])
            S = S + a * G
            self.alphas.append(a)
            self.stumps.append(stump)
            costs.append(cost)
            iter += 1
        return costs

    def predict(self, x):
        ret = 0.0
        for s, a in zip(self.stumps, self.alphas):
            ret += a * s.predict(x)
        return ret


if __name__ == "__main__":
    # x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1, 1)
    # y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

    x = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = np.sin(x).reshape(1, -1)[0]
    gb = GradientBoost(x, y)
    cost = gb.fit(400)
    pred = [gb.predict(x_) for x_ in x]
    print(pred)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(x, y)
    ax2 = fig.add_subplot(312)
    ax2.plot(x, pred)

    ax3 = fig.add_subplot(313)
    ax3.plot(cost)
    plt.show()
