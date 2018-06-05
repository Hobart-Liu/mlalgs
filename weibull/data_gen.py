import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import weibull_min


def weibull_data():

    p1 = weibull_min.rvs(0.6, loc=0, scale=2, size=5000)
    p2 = np.arange(1, 2000)
    lower, upper, mu, sigma = 1800, 2000, 1950, 50

    model = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    p3 = model.rvs(5000)
    return np.concatenate((p1, p2,  p3))




def generate_data():

    start, end = 0, 2000

    # even distribution

    p1 = np.random.uniform(start, end, 1000)

    # exponential distribution

    p2 = np.random.exponential(scale=50, size=20000)

    # truncated_normal

    lower, upper, mu, sigma = 1800, 2000, 1950, 50

    model = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    p3 = model.rvs(20000)

    return np.concatenate((p1, p2, p3))


def store_generated_data(points):
    np.savetxt('data1.csv', points, delimiter=',')
    return points


def load_data():
    points = np.loadtxt('data.csv',delimiter=',')
    return points


# points = load_data()

# points = generate_data()
# points = store_generated_data()

points = weibull_data()
store_generated_data(points)
print(points.shape)

sns.distplot(points, bins=100, kde=False)
plt.show()

