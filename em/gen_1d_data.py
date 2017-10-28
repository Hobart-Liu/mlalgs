import numpy as np
import pickle

mu1, std1 = 3.0, 1.0
mu2, std2 = 4.0, 1.5
nb_points = 5
data1 = np.random.normal(mu1, std1, nb_points)
data2 = np.random.normal(mu2, std2, nb_points)
X = np.hstack((data1, data2))
y = np.hstack((
    np.array([0] * nb_points), np.array([1] * nb_points)
))

idx = np.arange(len(X))
np.random.shuffle(idx)
X=X[idx]
y=y[idx]

d = dict(X=X, y=y)
with open('data1.pickle', 'wb') as f:
    pickle.dump(d, f)

with open('data1.pickle','rb') as f:
    new_d = pickle.load(f)
    new_X = new_d['X']
    new_y = new_d['y']


import matplotlib.pyplot as plt

def normpdf(x, mu, std):
    var = float(std)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mu))**2/(2*var))
    return num/denom

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


colors = ['#ebe4c2','#00a7bd']
for x, k in zip(X, y):
    ax1.scatter(x, 0, s=30, edgecolors='#646464', color=colors[k])

x1 = np.linspace(mu1 - 4 * std1, mu1 + 4 * std1, 100)
prob = np.array([normpdf(x_, mu1, std1) for x_ in x1])
ax1.plot(x1, prob, color=colors[0])
x2 = np.linspace(mu2 - 4 * std2, mu2 + 4 * std2, 100)
prob = np.array([normpdf(x_, mu2, std2) for x_ in x2])
ax1.plot(x2, prob, color=colors[1])


for x, k in zip(new_X, new_y):
    ax2.scatter(x, 0, s=30, edgecolors='#646464', color=colors[k])

x1 = np.linspace(mu1 - 4 * std1, mu1 + 4 * std1, 100)
prob = np.array([normpdf(x_, mu1, std1) for x_ in x1])
ax2.plot(x1, prob, color=colors[0])
x2 = np.linspace(mu2 - 4 * std2, mu2 + 4 * std2, 100)
prob = np.array([normpdf(x_, mu2, std2) for x_ in x2])
ax2.plot(x2, prob, color=colors[1])

print(new_X.shape)
print(new_y.shape)


plt.show()

