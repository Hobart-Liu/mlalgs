"""
My first EM practice.
It works sometime, but it is not stable. It counters dived by zero very often.
I suppose, in M step, when we update mu, the mu:s happened to be the same as one of the x:s

"""


import numpy as np
import pickle
from matplotlib.mlab import normpdf



#
# def normpdf(x, mu, std):
#     var = float(std)**2
#     denom = (2*np.pi*var)**.5
#     num = np.exp(-(float(x)-float(mu))**2/(2*var))
#     return num/denom

def fit_em(x, k, max_iter=200):
    m = len(x)
    # mu = np.array([2.57039876, 4.76793824])
    mu = np.random.choice(x, 2)
    print("random pick mu", mu)
    sigma = np.array([1.0] * k)
    w = [1./k]*k

    P = np.zeros((m, k))

    iter = 0

    old_L = 1000

    while iter < max_iter:
        iter += 1

        # E step
        for i in range(m):
            for j in range(k):
                P[i, j] = normpdf(x[i], mu[j], sigma[j]) * w[j]

        denom = P.sum(axis=1).reshape(-1, 1)
        P = P/denom


        # M step
        div = np.sum(P, axis=0)
        if (div==0).any():
            print(div)
        w = 1. / m * div

        for j in range(k):

            mu[j] = 1./div[j] * np.sum(P[:, j] * x)

            x_ = x - mu[j]
            sigma[j] = 0.
            for i in range(m):
                sigma[j] += P[i,j]*x_[i]*x_[i]
            sigma[j] /= div[j]

        # Likelihood
        L = 0.0
        for i in range(m):
            tmp = 0.0
            for j in range(k):
                tmp += w[j]*normpdf(x[i], mu[j], sigma[j])
            L += np.log(tmp)


        # print("Iter = ", iter)
        # print("mu = ", mu)
        # print("sigma = ", sigma)
        # print("L = ", L)
        # print("\n")
        if abs(L-old_L) < 1e-3: break
        old_L = L


    return mu, sigma


# with open('data1.pickle', 'rb') as f:
#     d = pickle.load(f)
#     X = d['X']
#     y = d['y']


mu1, std1 = 3.0, 1.0
mu2, std2 = 5.0, 1.5
num_points = 5
data1 = np.random.normal(mu1, std1, num_points)
data2 = np.random.normal(mu2, std2, num_points)
X = np.hstack((data1, data2))
y = np.hstack((np.array([0] * num_points), np.array([1] * num_points)))

idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

mu, sigma = fit_em(X, 2)
print(mu)
print(sigma)

import matplotlib.pyplot as plt
fig = plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)

colors = ['#ebe4c2','#00a7bd']

for x, k in zip(X, y):
    ax1.scatter(x, 0, s=30, edgecolors='#646464', color=colors[k])

x1 = np.linspace(mu1 - 4 * std1, mu1 + 4 * std1, 100)
prob = np.array([normpdf(x_, mu1, std1) for x_ in x1])
ax1.plot(x1, prob, color=colors[0])
x2 = np.linspace(mu2 - 4 * std2, mu2 + 4 * std2, 100)
prob = np.array([normpdf(x_, mu2, std2) for x_ in x2])
ax1.plot(x2, prob, color=colors[1])



for x, k in zip(X, y):
    ax2.scatter(x, 0, s=30, edgecolors='#646464', color=colors[k])

for c, m, s in zip(colors, mu, sigma):
    xx = np.linspace(m - 4* s, m+4*s, 100)
    prob = np.array([normpdf(x_, m, s) for x_ in xx])
    ax2.plot(xx, prob, color=c)
plt.show()