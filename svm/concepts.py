import numpy as np
from cvxopt import matrix, solvers

def distance(w, b, x, y):
    return y * (np.dot(w, x) + b)/np.linalg.norm(w)


def test_distance():
    x = np.array([[1, 1], [8, 1], [6, 6]])
    w = np.array([1, 1])
    b = -10
    y = np.array([-1,-1, 1])
    print(distance(w, b, x.T, y))

"""
[ 5.65685425  0.70710678  1.41421356]
"""

def qp1():

    # https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf

    """
    min x^2/2 + 3x + 4y
    x, y >= 0
    x+3y >= 15
    2x+5y <=100
    3x+4y <=80
    """

    Q = matrix([[1.0, 0.0], [0.0, 0.0]])
    p = matrix([3.0, 4.0])
    G = matrix([[-1.0, 0.0, -1.0, 2.0, 3.0], [0.0, -1.0, -3.0, 5.0, 4.0]])
    h = matrix([0.0, 0.0, -15.0, 100.0, 80.0])

    ans = solvers.qp(Q, p, G, h)
    print(ans['status'])
    print(ans['x'])
    print(ans['primal objective'])


def qp2():

    # http://cvxopt.org/examples/tutorial/qp.html

    """
    min 2x^2 + y^2 + xy + x + y
    s.t. x >= 0
         y >= 0
         x+y = 1

    """

    Q = matrix([[2.0, 0.5], [0.5, 1]])
    p = matrix([1.0, 1.0])
    G = matrix([[-1.0, 0.0], [0.0, -1.0]])
    h = matrix([0.0, 0.0])
    A = matrix([1.0, 1.0], (1, 2))
    b = matrix(1.0)

    ans = solvers.qp(Q, p, G, h, A, b)
    print(ans['status'])
    print(ans['x'])
    print(ans['primal objective'])


def min_alpha():
    """
    given X, cal min alpha
    min_a sum_ij_1_to_N(ai * aj * yi * yj *( xi * xj)) - sum_i_1_to_N(ai)
    s.t. sum_i_1_to_N(aiyi) = 0
         ai >= 0

    """
    x = [[3,3], [4,3], [1,1]]
    y = [1, 1, -1]






    pass

if __name__ == '__main__':
    qp1()





