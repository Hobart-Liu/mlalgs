"""
Regression Tree
CART approach
Label 是连续数据
训练数据是True/False --> 1.0/0.0

"""


import numpy as np
import pandas as pd
from collections import Counter

def mse(d, m):
    return sum([(x-m)**2 for x in d])

class Node(object):
    def __init__(self, col, split_point, result, left_branch, right_branch, sample):
        self.col = col # column name of criteria being tested
        self.split_point = split_point  # value of criteria
        self.results = result # endpoint otherwise none
        self.left_branch = left_branch # if value(col) < split_point
        self.right_branch = right_branch # otherwise
        self.sample = sample


def find_best_split(df, features, label_col):

    the_min = np.inf
    the_feat = None
    the_val = None
    the_left = None
    the_right = None

    for feat in features:
        for val in df[feat]:
            subdf1 = df.query(feat + '< @val')
            subdf2 = df.query(feat + '>= @val')
            c1 = subdf1[label_col].mean()
            c2 = subdf2[label_col].mean()
            mean_sequare_error = mse(subdf1[label_col], c1) + mse(subdf2[label_col], c2)
            if mean_sequare_error < the_min:
                the_min = mean_sequare_error
                the_feat = feat
                the_val = val
                the_left = subdf1
                the_right = subdf2

    return the_feat, the_val, the_left, the_right

def build_tree_CART(df, features, label_col, max_depth, min_size, depth):
    # print("\t"*depth, features)

    feat, split_point, df_l, df_r = find_best_split(df, features, label_col)

    Ck = Counter(df[label_col])
    if len(Ck.keys()) == 1:
        return Node(col=None, split_point=None, result=list(Ck.keys())[0],
                    left_branch=None, right_branch=None, sample=len(df))

    if len(features) == 0 or depth > max_depth or len(df) <= min_size:
        return Node(col=None, split_point=None, result=df[label_col].mean(),
                    left_branch=None, right_branch=None, sample=len(df))

    """
       Age  LikesGardening  PlaysVideoGames  LikesHats
    1   14             0.0              1.0        0.0
    2   15             0.0              1.0        0.0
    """
    if len(df_l) == 0 or len(df_r) == 0:
        return Node(col=None, split_point=None, result=df[label_col].mean(),
                    left_branch=None, right_branch=None, sample=len(df))

    features = features - set([feat])
    return Node(col=feat, split_point=split_point, result=None,
                left_branch=build_tree_CART(df_l, features, label_col, max_depth, min_size, depth + 1),
                right_branch=build_tree_CART(df_r, features, label_col, max_depth, min_size, depth + 1),
                sample=len(df)
                )




def printTree(tree, depth=0):
    if tree is None: return
    if tree.col is not None:
        print("\t"*depth, "{0} < {1}, sample={2}".format(tree.col, tree.split_point, tree.sample))
    if tree.results is None:
        printTree(tree.left_branch, depth+1)
        printTree(tree.right_branch, depth+1)
    else:
        print("\t"*depth, "Class is", tree.results)

if __name__ == "__main__":
    
    FALSE, TRUE = 0.0, 1.0
    data = [
        dict(Age=13, LikesGardening=FALSE, PlaysVideoGames=TRUE,  LikesHats=TRUE ),
        dict(Age=14, LikesGardening=FALSE, PlaysVideoGames=TRUE,  LikesHats=FALSE),
        dict(Age=15, LikesGardening=FALSE, PlaysVideoGames=TRUE,  LikesHats=FALSE),
        dict(Age=25, LikesGardening=TRUE,  PlaysVideoGames=TRUE,  LikesHats=TRUE ),
        dict(Age=35, LikesGardening=FALSE, PlaysVideoGames=TRUE,  LikesHats=TRUE ),
        dict(Age=49, LikesGardening=TRUE,  PlaysVideoGames=FALSE, LikesHats=FALSE),
        dict(Age=68, LikesGardening=TRUE,  PlaysVideoGames=TRUE,  LikesHats=TRUE ),
        dict(Age=71, LikesGardening=TRUE,  PlaysVideoGames=FALSE, LikesHats=FALSE),
        dict(Age=73, LikesGardening=TRUE,  PlaysVideoGames=FALSE, LikesHats=TRUE )

    ]

    df = pd.DataFrame(data, columns=['Age', 'LikesGardening', 'PlaysVideoGames', 'LikesHats'])
    print(df)

    tree = build_tree_CART(df, set(['LikesGardening', 'PlaysVideoGames', 'LikesHats']), 'Age',
                           np.inf, 3, 0)
    printTree(tree, 0)
