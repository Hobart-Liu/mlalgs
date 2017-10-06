def distance(xList, yList, p=1):
    """
    Compuate distance between vectors
    p = 1, Manhattan distance
    p = 2, Euclidean distance
    p = inf, not supported
    参见李航，统计学习方法， P39
    """
    assert(isinstance(xList, list))
    assert(isinstance(yList, list))
    assert(p != 0)

    dist = [abs(a-b)**p for a, b in zip(xList, yList)]
    dist = sum(dist)**(1/p)

    return dist


x = [8, 8, 10, 30, 20, 12, 6, -1]
y = [8, 7, 6, 5, 4, 3, 2, 1]

print(distance(x, y, 1))
print(distance(x, y, 0.5))

x1 = [1,1]
x2 = [5,1]
x3 = [4,4]

print(distance(x1, x3, 1))
print(distance(x1, x3, 2))
print(distance(x1, x3, 3))
print(distance(x1, x3, 4))


