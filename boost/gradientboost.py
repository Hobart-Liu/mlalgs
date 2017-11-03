"""
Build CART stump, with one decision branch

"""


import numpy as np
import matplotlib.pyplot as plt


def mse(d, m):
    return sum([(x - m) ** 2 for x in d])

class CARTStump(object):
    def __init__(self):
        self.split_point = None
        self.feat = None
        self.less_than_result = None  # <=
        self.larger_than_result = None # >

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
            for row in range(m-1):
                split_point = X[row, feat]
                sub1 = Y[data <= split_point]
                sub2 = Y[data > split_point]
                y_hat_1 = sub1.mean()
                y_hat_2 = sub2.mean()
                mean_sequre_error= mse(sub1, y_hat_1) + mse(sub2, y_hat_2)
                if mean_sequre_error < the_min:
                    the_min = mean_sequre_error
                    self.split_point = split_point
                    self.feat = feat
                    self.less_than_result = y_hat_1
                    self.larger_than_result = y_hat_2

    def predict(self, x):
        if x[self.feat] <= self.split_point:
            return self.less_than_result
        else:
            return self.larger_than_result

class GradientBoost:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.S = []

    def fit(self, max_iter = 5):
        iter = 0
        cost = []
        residual = self.Y
        while iter < max_iter:
            print("===== Iter", iter, end="")
            stump = CARTStump()
            stump.fit(self.X, residual)
            new_r = np.array([r - stump.predict(x) for r, x in zip(residual, self.X)])
            residual = new_r
            cost.append(sum([r**2 for r in residual]))
            print(cost[-1])
            self.S.append(stump)
            iter += 1
        return cost

    def predict(self, x):
        ret = 0.0
        for s in self.S:
            ret += s.predict(x)
        return ret



if __name__ == "__main__":
    # x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1, 1)
    # y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

    x = np.linspace(-10, 10, 1000).reshape(-1, 1)
    y = np.sin(x)
    gb = GradientBoost(x, y)
    cost = gb.fit(50)
    pred = [gb.predict(x_) for x_ in x]
    print(pred)

    fig=plt.figure()
    ax1=fig.add_subplot(311)
    ax1.plot(x, y)
    ax2=fig.add_subplot(312)
    ax2.plot(x, pred)

    ax3=fig.add_subplot(313)
    ax3.plot(cost)
    plt.show()




