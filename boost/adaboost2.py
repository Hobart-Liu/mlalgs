import numpy as np

LESSTHAN = 0
LARGERTHAN = 1
disp = ['<=', ">"]

class Stump:
    """
    Simple Stump
    use > or <= as classifier
    """
    def __init__(self, X, Y, weight):
        self.split = None
        self.func = None
        self.errors = None
        self.err = None
        besterr = np.inf



        for x in X:
            predict = [self.less_than(x_, x) for x_ in X]
            errors = [p != y for p, y in zip(predict, Y)]
            err = sum(weight*errors)
            if err < besterr:
                self.split = x
                self.func = self.less_than
                self.errors = errors
                self.err = err
                besterr = err
                print("err = {}, {} {}".format(err, disp[0], x))

            predict = [self.larger_than(x_, x) for x_ in X]
            errors = [p != y for p, y in zip(predict, Y)]
            err = sum(weight*errors)
            if err < besterr:
                self.split = x
                self.func = self.larger_than
                self.errors = errors
                self.err = err
                besterr = err
                print("err = {}, {} {}".format(err, disp[1], x))



    def larger_than(self, x, y):
        if x > y : return  1
        else     : return -1

    def less_than(self, x, y):
        if x <= y: return  1
        else     : return -1


    def pred(self, x):
        return self.func(x, self.split)

class AdaBoost:

    def __init__(self, dataset, label):
        self.X = dataset # train dataset
        self.Y = label # label
        self.N = len(dataset) # len
        self.W = np.ones(self.N)/self.N # weights
        self.A = [] # alphas
        self.S = [] # stumps

    def fit(self, max_iter=10):
        iter = 0
        while iter < max_iter:
            print("Iter, ",  iter, "=========")
            stump = Stump(self.X, self.Y, self.W)
            e = stump.err
            a = np.log((1-e)/e) * 0.5
            w = np.zeros(self.N)
            for i, (wi, erri) in enumerate(zip(self.W, stump.errors)):
                if erri: w[i] = wi * np.exp( a)
                else   : w[i] = wi * np.exp(-a)
            self.W = w / w.sum()
            self.S.append(stump)
            self.A.append(a)
            iter += 1

    def evaluate(self):
        count = 0
        for x, y in zip(self.X, self.Y):
            hx = [a*s.pred(x) for a, s in zip(self.A, self.S)]
            pred = np.sign(sum(hx))
            if pred == y: count += 1
            print("data x %+4d, label %+2d, predict %+2d" %(x, y,np.sign(sum(hx))))
        print("accuracy", count/self.N*100.0)

if __name__ == "__main__":
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    y = np.array([1, 1, 1,-1,-1,-1, 1, 1, 1,-1])

    boost = AdaBoost(x, y)
    boost.fit(3)
    boost.evaluate()

    x = np.random.randint(-100, 100, 50)
    y = np.random.choice([+1, -1], 50)
    boost = AdaBoost(x, y)
    boost.fit(20)
    boost.evaluate()


