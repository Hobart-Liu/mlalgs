from operator import itemgetter

# reference: https://en.wikipedia.org/wiki/K-d_tree
# reference: https://salzis.wordpress.com/2014/06/28/kd-tree-and-nearest-neighbor-nn-search-2d-case/


class Node:
    def __init__(self, point, left_child, right_child):
        self.point = point
        self.left_child = left_child
        self.right_child = right_child



def kdtree(point_list, depth):
    """caller has to ensure the point_list has same dimension for each element"""

    try:
        k = len(point_list[0])
    except IndexError:
        return None

    axis = depth % k

    point_list.sort(key=itemgetter(axis))
    median = len(point_list)//2
    # say point_list has 7 elements
    # median = 3, left= 0, 1, 2, right = 4, 5, 6
    # say point_list has 6 elements
    # median = 3, left= 0, 1, 2, right = 4, 5

    print("axis = ", axis, "point = ", point_list[median])
    print(point_list)

    return Node(point=point_list[median],
                left_child=kdtree(point_list[:median], depth+1),
                right_child=kdtree(point_list[median+1:], depth+1)
                )


"""
Below is testing and intuition
"""

from random import randint
import numpy as np
import matplotlib.pyplot as plt


def print_tree(tree, depth):
    if tree is None:
        print("\t"*depth, "None")
        return
    print("\t"*depth, tree.point)
    print_tree(tree.left_child, depth+1)
    print_tree(tree.right_child, depth+1)

def generate_point_list(n, min_val, max_val):
    return [[randint(min_val, max_val), randint(min_val, max_val)] for _ in range(n)]

n = 10
min_val = 0
max_val = 20
x_list = generate_point_list(n, min_val, max_val)
tree = kdtree(x_list, 0)
print_tree(tree, 0)

line_width = [4., 3.5, 3., 2.5, 2., 1.5, 1., .5, 0.3]

def plot_tree(tree, min_x, max_x, min_y, max_y, prev_node, branch, depth=0):

    point = tree.point
    left_branch=tree.left_child
    right_branch=tree.right_child

    ln_width = line_width[
        min(depth, len(line_width)-1)
    ]

    k = len(point)
    axis = depth % k

    # draw a vertical splitting line
    if axis==0:
        if branch is not None and prev_node is not None:
            if branch:
                max_y = prev_node[1]
            else:
                min_y = prev_node[1]
        plt.plot([point[0], point[0]], [min_y, max_y],
                 linestyle='-', linewidth=ln_width,
                 color='r')
    # draw a horizontal splitting line
    elif axis==1:
        if branch is not None and prev_node is not None:
            if branch:
                max_x=prev_node[0]
            else:
                min_x=prev_node[0]
        plt.plot([min_x, max_x], [point[1], point[1]],
                 linestyle='-', linewidth=ln_width,
                 color='b')

    # draw the current point
    plt.plot(point[0], point[1], 'ko')

    # draw left and right branches of the current point
    if left_branch is not None:
        plot_tree(left_branch, min_x, max_x, min_y, max_y, point, True, depth+1)
    if right_branch is not None:
        plot_tree(right_branch, min_x, max_x, min_y, max_y, point, False, depth+1)

delta = 5
plt.figure("K-d Tree", figsize=(10., 10.))
plt.axis([min_val-delta, max_val+delta, min_val-delta, max_val+delta])
plt.grid(b=True, which='major', color='0.75', linestyle='--')
plt.xticks([i for i in range(min_val - delta, max_val + delta, 1)])
plt.yticks([i for i in range(min_val - delta, max_val + delta, 1)])

# draw the tree
plot_tree(tree, min_val - delta, max_val + delta, min_val - delta, max_val + delta, None, None)

plt.title('K-D Tree')
plt.show()
plt.close()

