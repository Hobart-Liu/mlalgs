import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cvxopt

"""
第一版 softmargin + kernel, support vector 基本都能找出来
分类的准确率大致在90%+
（TODO：做分割线）
"""


def generate_samples(N, min=0.0, max=10.0):
    width = max - min
    scale = 1.0
    loc1 = [width*1/5, width*1/5]
    loc2 = [width*3/5, width*3/5]
    loc3 = [width*1/5, width*3/5]
    loc4 = [width*3/5, width*1/5]
    ds1 = np.random.normal(loc=loc1, scale=scale, size=(N,2))
    ds2 = np.random.normal(loc=loc2, scale=scale, size=(N,2))
    ds3 = np.random.normal(loc=loc3, scale=scale, size=(N,2))
    ds4 = np.random.normal(loc=loc4, scale=scale, size=(N,2))
    y1 = [-1]*N
    y2 = [1]*N
    y = np.vstack((np.array(y1)[:, None], np.array(y2)[:, None], np.array(y2)[:, None], np.array(y2)[:, None]))
    ds = np.hstack((np.vstack((ds1, ds2, ds3, ds4)), y))
    np.random.shuffle(ds)
    return ds[:N, :2], ds[:N, 2], ds[N:, :2], ds[N:, 2]

def plot_point(ds, lbl):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for point, y in zip(ds, lbl):
        color = 'r' if y==1 else 'b'
        marker = 'o' if y==1 else '^'
        ax.scatter(point[0], point[1], s = 10, marker=marker, c=color)
    plt.show()

# define kernels
def linear():
    return lambda x, y: np.inner(x, y)

def gaussian(sigma):
    return lambda x, y: np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

class SVM(object):
    def __init__(self):
        self.sv_points = None
        self.sv_labels = None
        self.alphas = None
        self.b_svm = None
        self.w_svm = None

    # define SVM functions
    def fit(self, data, label, kernel, C = 999999999.0):
        l = len(data)
        K = np.zeros(shape=(l, l))
        for i in range(l):
            for j in range(l):
                K[i,j] = kernel(data[i], data[j])

        P = cvxopt.matrix(np.outer(label, label) * K)
        q = cvxopt.matrix(np.ones(l) * -1)
        G = cvxopt.matrix(np.vstack((
            np.eye(l) * -1,
            np.eye(l)
        )))

        h = cvxopt.matrix(np.hstack((
            np.zeros(l),
            np.ones(l) * C
        )))

        A = cvxopt.matrix(label, (1, l))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P,q, G, h, A, b)

        a = np.ravel(solution['x'])
        index = a > 1e-5
        self.alphas = a[index]
        self.sv_points = data[index]
        self.sv_labels = label[index]

        self.w_svm = np.zeros(data.shape[1])

        for i in range(len(self.alphas)):
            self.w_svm += self.alphas[i] * self.sv_labels[i] * self.sv_points[i]

        self.b_svm = self.sv_labels[0]
        for i in range(len(self.alphas)):
            self.b_svm -= self.sv_labels[i] * self.alphas[i] * kernel(self.sv_points[i], self.sv_points[0])

        return self.sv_points

    def predict(self, point, kernel):
        ret = self.b_svm
        for i in range(len(self.alphas)):
            ret += self.alphas[i] * self.sv_labels[i]* kernel(self.sv_points[i], point)
        return ret


if __name__ == '__main__':
    x, y, test_x, test_y = generate_samples(100)
    # plot_point(x, y)

    svm = SVM()
    f = linear()

    sv= svm.fit(x, y, linear(), C=50)
    print(len(sv))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for point, y_ in zip(x, y):
        color = 'r' if y_==1 else 'b'
        marker = 'o' if y_==1 else '^'
        ax.scatter(point[0], point[1], s = 15, marker=marker, c=color)

    for point in sv:
        ax.scatter(point[0], point[1], facecolor='none', s=200, edgecolors='k')

    count = 0
    for p, l in zip(test_x, test_y):
        pred = np.sign(svm.predict(p, f))
        print("point {}, predict {}, label {}".format(p, pred, l))
        if pred == l: count += 1

    print("Percentage", count/len(test_x))
    plt.show()