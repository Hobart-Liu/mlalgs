import matplotlib.pyplot as plt
import numpy as np


def H(p):
    if p in [0, 1]:
        return 0
    else:
        return -p * np.log(p)/np.log(2) - (1-p) * np.log(1-p)/np.log(2)

# build (0, 1)
x = np.linspace(0, 1, 1012)
x = np.delete(x, 0)
x = np.delete(x, -1)

y = [H(p) for p in x]

# print(x)
# print(y)

plt.plot(x, y, '-')
plt.xlabel("p")
plt.ylabel("empirical_entropy(p)")
plt.grid(True)
plt.show()