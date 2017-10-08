from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np

class KDTree(object):

    Manhattan = 1
    Euclidean = 2

    def __init__(self, x):
        assert(isinstance(x, list))
        assert(isinstance(x[0], list))

        self.tree = self.__build_kdtree(x)

    def __build_kdtree(self, point_list, depth=0):
        try:
            k = len(point_list[0])
        except IndexError:
            return None

        axis = depth % k

        point_list.sort(key=itemgetter(axis))
        median = len(point_list) // 2

        return dict(point=point_list[median],
                    left = self.__build_kdtree(point_list[:median], depth + 1),
                    right= self.__build_kdtree(point_list[median+1:], depth + 1))


    def closest_point(self, x, distance_type):
        return self.__recursive_search(self.tree, x, distance_type, depth=0)


    def __recursive_search(self, node, x, distance_type, depth=0):
        print("\t"*depth, "Entering with node", node['point'] if node is not None else None)
        if node is None: return None, np.inf
        k = len(node['point'])
        axis = depth % k

        distance1 = self.__distance(node['point'], x, distance_type)
        print("\t"*depth, "distance=", distance1)

        if x[axis] < node['point'][axis]:
            next_branch = node['left']
            opposite_branch = node['right']
        else:
            next_branch = node['right']
            opposite_branch = node['left']


        next_node, distance2 = self.__recursive_search(next_branch, x, distance_type, depth + 1)
        best = node if distance1 <= distance2 else next_node
        distance = min(distance1, distance2)
        print("\t"*depth, "next_node", next_node['point'] if next_node is not None else None, "distance=", distance2)
        print("\t"*depth, "Best node", best['point'], "Distance = ", distance)

        if distance > abs(x[axis] - node['point'][axis]):
            print("\t"*depth, "intersect", "distance=", distance, "axis=", axis, "x=", x, "node=", node['point'])
            next_node, distance2 = self.__recursive_search(opposite_branch, x, distance_type, depth +1)
            best = node if distance1 <= distance2 else next_node
            distance = min(distance1, distance2)


        return best, distance

    def __distance(self, xlist, ylist, p=1):
        """
        compute distance between 2 vector lists

        p = 1: Manhattan distance
        p = 2: Euclidean distance
        ...
        p = inf, not support
        参见李航，统计学习方法， P39
        """
        assert(isinstance(xlist, list))
        assert(isinstance(ylist, list))
        assert(p != 0)

        dist = [abs(a-b)**p for a, b in zip(xlist, ylist)]
        dist = sum(dist)**(1/p)

        return dist

    def print_tree(self):
        self.__recursive_print(self.tree)

    def __recursive_print(self, tree, depth=0):
        if tree is None:
            print("\t" * depth, "None")
            return
        print("\t" * depth, tree['point'])
        self.__recursive_print(tree['left'], depth + 1)
        self.__recursive_print(tree['right'], depth + 1)

    def plot_tree(self, min_x, max_x, min_y, max_y):
        self.__recursive_plot(self.tree, min_x, max_x, min_y, max_y)
        plt.show()

    def __recursive_plot(self, tree, min_x, max_x, min_y, max_y, prev_node=None, branch=None, depth=0):


        line_width = [4., 3.5, 3., 2.5, 2., 1.5, 1., .5, 0.3]

        point = tree['point']
        left_branch=tree['left']
        right_branch=tree['right']

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
            self.__recursive_plot(left_branch, min_x, max_x, min_y, max_y, point, True, depth+1)
        if right_branch is not None:
            self.__recursive_plot(right_branch, min_x, max_x, min_y, max_y, point, False, depth+1)



point_list = [[50, 20], [20, 40], [80, 70], [10, 10], [45, 80], [70, 90], [90, 10]]
pivot = [55, 80]

tree = KDTree(point_list)
# KDTree.print_tree(tree)
# KDTree.plot_tree(tree, 0, 100, 0, 100)
print(tree.closest_point(pivot, KDTree.Euclidean))









