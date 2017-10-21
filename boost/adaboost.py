"""
该adaboost 是参考李航 统计学习方法 P138页算法实现。具体的数据例子,以及中间计算数据参见p141
参考了：https://gist.github.com/tristanwietsma/5486024的实现

另：
1. 关于弱分类器如何迭代，这里没做考虑，留到rf的实现
2. 按MIT机器学习课程，Patrick Winston,似乎在权重调整时有不同的算法，将代码实现赋值如下，留待以后再讨论。
https://www.youtube.com/watch?v=UHBmv7qCey4

        w = np.zeros(self.N)
        sRight = np.sum(self.weights[errors])
        sWrong = np.sum(self.weights[~errors])
        for i in range(self.N):
            if errors[i] == 1: w[i] = self.weights[i]/(2.0*sRight)
            else: w[i] = self.weights[i]/(2.0*sWrong)

        self.weights = w / w.sum()


"""

import numpy as np

class AdaBoost:

    def __init__(self, dataset):
        self.training_set = dataset
        self.N = len(dataset)
        self.weights = np.ones(self.N)/self.N
        self.rules = []
        self.alphas = []

    def set_rule(self, func):
        errors = np.array([t[1] != func(t[0]) for t in self.training_set])
        tmp = np.array([func(t[0]) for t in self.training_set])
        print(tmp)
        print(self.training_set[:, 1])
        print(errors)
        e = (errors*self.weights).sum()
        alpha = np.log((1-e)/e) * 0.5
        print("e=%.2f, a=%.2f"%(e, alpha))
        w = np.zeros(self.N)
        for i, (wi, erri) in enumerate(zip(self.weights, errors)):
            if erri: w[i] = wi * np.exp( alpha)
            else   : w[i] = wi * np.exp(-alpha)
        self.weights = w / w.sum()
        print(self.weights)
        self.rules.append(func)
        self.alphas.append(alpha)

    def evaluate(self):
        for (x, l) in self.training_set:
            hx = [alpha * func(x) for alpha, func in zip(self.alphas, self.rules)]
            print("data ", x, "label ", l, "predict", np.sign(sum(hx)))






if __name__ == "__main__":
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    y = np.array([1, 1, 1,-1,-1,-1, 1, 1, 1,-1])
    train_data = np.column_stack((x, y))

    boost = AdaBoost(train_data)
    boost.set_rule(lambda x: 1 if x < 2.5 else -1)
    boost.set_rule(lambda x: 1 if x < 8.5 else -1)
    boost.set_rule(lambda x: 1 if x > 5.5 else -1)

    boost.evaluate()


