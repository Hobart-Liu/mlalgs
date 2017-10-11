'''
Show all different interpolation methods for imshow
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.animation as animation

WEIGHT_DUMP = 'weight.dump'

fig1, axes1 = plt.subplots(10, 8, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})
fig1.subplots_adjust(hspace=0.3, wspace=0.05)




def animate(t):
    with open(WEIGHT_DUMP, 'rb') as f:
        try:
            d = pickle.load(f)
            w1 = d.get('w1')
            w2 = d.get('w2')

            for i in range(7):
                for j in range(7):
                    axes1[i, j].imshow(w1[i * 7 + j].reshape(28, 28), cmap='gray')

            for i in range(10):
                axes1[i,7].imshow(w2[i].reshape(7,7), cmap='gray')

        except EOFError:
            print("skip EOF Error")


ani = animation.FuncAnimation(fig1, animate, interval= 1000)

plt.show()
