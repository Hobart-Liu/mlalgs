'''
Show all different interpolation methods for imshow
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.animation as animation
from matplotlib import style

#style.use('dark_background')

WEIGHT_DUMP = 'weight_2.dump'

fig1, axes1 = plt.subplots(10, 20, figsize=(15, 15),
                           subplot_kw={'xticks': [], 'yticks': []},)
fig1.subplots_adjust(hspace=0.3, wspace=0.05)
#fig1.subplots_adjust(left = 0.01, bottom = 0.01, right = 0.01, top = 0.01, hspace=0.3, wspace=0.05)
for i in range(10):
    for j in range(20):
        axes1[i,j].axis('off')

def animate(t):
    with open(WEIGHT_DUMP, 'rb') as f:
        try:
            d = pickle.load(f)
            w1 = d.get('w1')
            w2 = d.get('w2')
            w3 = d.get('w3')


            for i in range(10):
                for j in range(10):
                    axes1[i, j].imshow(w1[i * 10 + j].reshape(28, 28), cmap='gray')

            for i in range(7):
                for j in range(7):
                    axes1[2+i, 11+j].imshow(w2[i * 7 + j].reshape(10, 10), cmap='gray')

            for i in range(10):
                axes1[i, 19].imshow(w3[i].reshape(7, 7), cmap='gray')

        except EOFError:
            print("skip EOF Error")


ani = animation.FuncAnimation(fig1, animate, interval= 5000)

plt.show()
