import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# very good reference:
# http://cs229.stanford.edu/section/gaussians.pdf
# http://cs229.stanford.edu/notes/cs229-notes7b.pdf
# https://gist.github.com/bistaumanga/6023716



def fit_em(x, k, max_iters=200):
    """

    :param x: [nxd] matrix, n, number of data points, d dimension of data points
    :param k:
    :param max_iter:
    :return:  list of mu and sigma of gaussian
    """
    n, d = x.shape

    mu = x[np.random.choice(n, k, replace=False), :]
    sigma = np.array([np.eye(d)] * k)
    w = np.array([1. / k] * k)
    R = np.zeros((n, k))

    L = - np.inf

    iter = 0
    while iter < max_iters:
        for k_ in range(k):
            R[:, k_] = w[k_] * multivariate_normal.pdf(x, mu[k_], sigma[k_])

        log_likelihood = np.sum(np.log(R.sum(axis=1)))
        R = R / (R.sum(axis=1).reshape(-1, 1))

        sum_w_j = R.sum(axis=0)

        for k_ in range(k):
            mu[k_] = 1./sum_w_j[k_] * np.dot(R[:, k_], x)
            x_mu = x - mu[k_]
            sigma[k_] = 1./sum_w_j[k_] * np.dot(np.multiply(R[:, k_], x_mu.T), x_mu)
            w[k_] = 1. / n * sum_w_j[k_]

        iter += 1
        if np.abs(log_likelihood - L) <= 1e-3: break
        L = log_likelihood

    return mu, sigma


##########################

# Demo part

if __name__ == "__main__":
    clusters = 4
    X, Y = make_blobs(n_samples=200, n_features=2, centers=clusters,
                      cluster_std=1.7, shuffle=True)
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1])

    mu, sigma = fit_em(X, clusters)

    px, py = np.mgrid[-15:15:0.1, -15:15:0.1]
    pos = np.empty(px.shape + (2,))
    pos[:,:,0]=px
    pos[:,:,1]=py

    prob = np.zeros((len(px), len(py)))
    for m, s in zip(mu, sigma):
        prob += multivariate_normal.pdf(pos, m, s)
    prob = prob/len(mu)
    cf = ax.contour(px, py, prob, 10, colors='k')
    ax.clabel(cf, inline=True, fontsize=10)



    plt.show()
