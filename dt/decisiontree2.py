
"""
CART approach
训练数据是连续值
"""

from os.path import join, expanduser
import pandas as pd
import numpy as np



# load a csv file
def load_data():
    filename = join(expanduser("~"), "mldata/misc/data_banknote_authentication.csv")
    df = pd.read_csv(filename, header=None, dtype=np.float32)
    return df.get_values()


def gini_index(groups, classes):
    """
    Calculate gini impurity
    :param groups: group instances
    :param classes: unique class instance
    :return: gini

    参见统计学习方法P69
    gini = |D1|/|D| * f(D1) + |D2|/|D| * f(D2)
    f(D) = 1 - sum_k((|C_k|/\D|) ** 2)

    C_k 是 D中属于第k类的样本子集， |C_k|是该子集的个数，|D|是数据集的个数
    D 根据特征 A 取某一个值a， 分割成D1, D2,

    """
    n_total_instances = float(
        sum(
            [len(group) for group in groups]
        )
    )

    gini = 0.0

    for group in groups:
        size = float(len(group))
        if size == 0: continue
        score = 0.0
        for class_val in classes:
            # row[-1] 是该实例的分类，
            # 计算C_k/D
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p

        gini += (1.0 - score) * (size / n_total_instances)

    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            try:
                left = np.vstack((left, row))
            except ValueError:
                left = [row]
        else:
            try:
                right = np.vstack((right, row))
            except ValueError:
                right = [row]

    return left, right


def get_split(dataset, disp=False):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = np.inf, np.inf, np.inf, None
    for index in range(len(dataset[0]) - 1):  # exclude last column, which is label
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            # computing gini is O(row*col*row)
            gini = gini_index(groups, class_values)
            if disp: print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], gini))
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])

    if len(left) == 0 or len(right) == 0:
        ds = right if len(left) == 0 else left
        node['left'] = node['right'] = to_terminal(ds)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right']=to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def tt1():
    assert (gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]) == 0.5)
    assert (gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]) == 0.0)
    dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.81281357, 0],
               [3.961043357, 2.61995032, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.00220326, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]
    split = get_split(dataset, True)
    print('Split: [X%d < %.3f]' % ((split['index'] + 1), split['value']))

"""
X1 < 2.771 Gini=0.444
X1 < 1.729 Gini=0.500
X1 < 3.678 Gini=0.286
X1 < 3.961 Gini=0.167
X1 < 2.999 Gini=0.375
X1 < 7.498 Gini=0.286
X1 < 9.002 Gini=0.375
X1 < 7.445 Gini=0.167
X1 < 10.125 Gini=0.444
X1 < 6.642 Gini=0.000
X2 < 1.785 Gini=0.500
X2 < 1.170 Gini=0.444
X2 < 2.813 Gini=0.320
X2 < 2.620 Gini=0.417
X2 < 2.209 Gini=0.476
X2 < 3.163 Gini=0.167
X2 < 3.339 Gini=0.444
X2 < 0.477 Gini=0.500
X2 < 3.235 Gini=0.286
X2 < 3.320 Gini=0.375
Split: [X1 < 6.642]
"""

def tt2():
    dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.81281357, 0],
               [3.961043357, 2.61995032, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.00220326, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]

    tree = build_tree(dataset, 1, 1)
    print_tree(tree)
    print("\n")
    tree = build_tree(dataset, 2, 1)
    print_tree(tree)
    print("\n")
    tree = build_tree(dataset, 3, 1)
    print_tree(tree)
    print("\n")
    tree = build_tree(dataset, np.inf, 1)
    print_tree(tree)

"""
[X1 < 6.642]
 [0.0]
 [1.0]


[X1 < 6.642]
 [X1 < 2.771]
  [0.0]
  [0.0]
 [X1 < 7.498]
  [1.0]
  [1.0]


[X1 < 6.642]
 [X1 < 2.771]
  [0.0]
  [X1 < 2.771]
   [0.0]
   [0.0]
 [X1 < 7.498]
  [X1 < 7.445]
   [1.0]
   [1.0]
  [X1 < 7.498]
   [1.0]
   [1.0]


[X1 < 6.642]
 [X1 < 2.771]
  [0.0]
  [X1 < 2.771]
   [0.0]
   [0.0]
 [X1 < 7.498]
  [X1 < 7.445]
   [1.0]
   [1.0]
  [X1 < 7.498]
   [1.0]
   [1.0]
   """

def tt3():
    dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.81281357, 0],
               [3.961043357, 2.61995032, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.00220326, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]

    stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
    for row in dataset:
        prediction = predict(stump, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))

"""
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=0, Got=0
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
Expected=1, Got=1
"""

def accuracy(actual, predicted):
    return sum(actual==predicted)/len(actual) * 100.0

def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


def evaluate_algorithm(dataset, n_folds, max_depth, min_size):
    nb_feature = len(dataset[0])
    scores = list()
    datasets = np.array_split(dataset, n_folds)
    idx = np.arange(len(datasets))
    for i in idx:
        train = None
        for n in np.delete(idx, i):
            try:
                train = np.vstack((train, datasets[n]))
            except ValueError:
                train = datasets[n]

        test = datasets[i]

        predicted = decision_tree(train, test[:, 0:nb_feature-1], max_depth, min_size)
        acc = accuracy(test[:, -1], predicted)
        print("round ", i, "acc=", acc)
        scores.append(acc)

    return scores




if __name__ == '__main__':

    data = load_data()
    scores = evaluate_algorithm(dataset=data, n_folds=5, max_depth=5, min_size=10)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


        # data = load_data()
        # assert(len(data) == 1372)
        # data = np.array_split(data, 10)
        # for i in range(len(data)):
        #     print(i, len(data[i]))

