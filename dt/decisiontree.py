import pandas as pd
import numpy as np
from collections import defaultdict, Counter

"""
限制：只针对离散值。
Open Question: 连续值如何处理
连续值一个可能的方法是将连续值排序后分成N个类，取类间的中位数为候选的划分点，
考察那个点带来的信息增益率最大（我们知道分类的情况， 反复调用info_gain_ratio）。
1）目前还没想到N如何确定
2）如果数据量很大，每个小区间的数据量都很大的话，是取中位数还是和概率密度有关，不确定。

"""

def get_freq_tbls(df, label_col='label'):
    freq_tbls = dict()

    label_values = df[label_col].unique()
    features = df.columns

    for feature in features:
        if feature == label_col: continue
        d = []
        feature_values = df[feature].unique()
        for feature_value in feature_values:
            d1 = []
            for label_value in label_values:
                d1.append(
                    len(
                        df.query(feature + ' == @feature_value and ' + label_col + '==@label_value')
                    )
                )
            d.append(d1)
        freq_tbls[feature] = pd.DataFrame(d, columns=label_values, index=feature_values)

    return freq_tbls

def empirical_entropy(df, label_col='label'):
    c = Counter(df[label_col])
    D = len(df)
    Ck_distribution = [x/D for x in c.values() if x!=0]
    return sum(
        [-p * np.log2(p) for p in Ck_distribution]
    )

def info_gain(freq_tbl, H_D):
    Di = freq_tbl.sum(axis=1)
    Di_D = Di/sum(Di)
    Dik_Di = freq_tbl.div(Di, axis=0)
    H_D_given_A = 0
    for i in range(len(freq_tbl)):
        tmp = [x * np.log2(x) for x in Dik_Di.iloc[i] if x != 0]
        H_D_given_A -= Di_D.iloc[i] * sum(tmp)

    return H_D-H_D_given_A

def info_gain_ratio(freq_tbl, H_D):
    gain = info_gain(freq_tbl, H_D)
    D = freq_tbl.values.sum()
    Di = freq_tbl.sum(axis=1)
    H_a_D = -sum(
        [x/D * np.log2(x/D) for x in Di]
    )
    if H_a_D == 0:
        print("H_a_D is zero")
    return gain/H_a_D



class Node(object):
    def __init__(self, col, results, branches):
        self.col = col # column name of criteria being tested
        self.results = results # endpoint otherwise none
        self.branches = branches

def build_tree_ID3(df, features, label_col, epsilon):
    """
    Build decision tree (ID3)
    :param df: train data set D
    :param features: feature set A, datatype Set
    :param label_col: label column name
    :param epsilon: threshold
    :return: decision tree
    """
    #1 若D中所有实例属于同一类Ck， 则T为叶节点，并将Ck作为该节点的类标记，返回
    Ck = Counter(df[label_col])
    if len(Ck.keys()) == 1:
        return Node(col=None, results=list(Ck.keys())[0], branches=None)

    #2 若features为空，则构建叶节点，以D中实例数最大的类Ck作为该节点的类标记，返回
    [(most_large_class, _)] = Ck.most_common(1)
    if len(features) == 0:
        return Node(col=None, results=most_large_class, branches=None)

    #3 计算features中各特征对D的信息增益，选择信息增益最大的max_info_gain_feature
    H_D = empirical_entropy(df, label_col=label_col)
    freq_tbls = get_freq_tbls(df, label_col=label_col)
    max_info_gain_feature = None
    max_info_gain = 0
    for feature, data in freq_tbls.items():
        infogain = info_gain(freq_tbls[feature], H_D)
        if infogain > max_info_gain:
            max_info_gain = infogain
            max_info_gain_feature = feature

    #4 如果信息增益小于epsilon，置T为叶节点，以D中实例数最大的类Ck作为该节点的类标记，返回
    if max_info_gain < epsilon:
        [(most_large_class, _)] = Ck.most_common(1)
        return Node(col=None, results=most_large_class, branches=None)

    #5 对max_info_gain_feature每一个可能的值v，构建若干个非空子集， 将实例数最大的类作为标记构建子节点
    #6 对第i个子节点，以Di为训练集，features-max_info_gain_feature为特征集， 递归调用1-5

    features = features - set([max_info_gain_feature])
    feature_values = df[max_info_gain_feature].unique()
    branches = dict()
    for value in feature_values:
        subdf = df.query(max_info_gain_feature + "==@value")
        branches[value] = build_tree_ID3(subdf, features, label_col, epsilon)

    return Node(col=max_info_gain_feature,
                results= most_large_class,
                branches=branches)


def build_tree_C45(df, features, label_col, epsilon):
    """
    Build decision tree (C4.5)
    :param df: train data set D
    :param features: feature set A, datatype Set
    :param label_col: label column name
    :param epsilon: threshold
    :return: decision tree
    """
    #1 若D中所有实例属于同一类Ck， 则T为叶节点，并将Ck作为该节点的类标记，返回
    Ck = Counter(df[label_col])
    if len(Ck.keys()) == 1:
        return Node(col=None, results=list(Ck.keys())[0], branches=None)

    #2 若features为空，则构建叶节点，以D中实例数最大的类Ck作为该节点的类标记，返回
    [(most_large_class, _)] = Ck.most_common(1)
    if len(features) == 0:
        return Node(col=None, results=most_large_class, branches=None)

    #3 计算features中各特征对D的信息增益，选择信息增益最大的max_info_gain_feature
    H_D = empirical_entropy(df, label_col=label_col)
    freq_tbls = get_freq_tbls(df, label_col=label_col)
    max_info_gain_feature = None
    max_info_gain = 0
    for feature, data in freq_tbls.items():
        infogain = info_gain_ratio(freq_tbls[feature], H_D)  # the only diff btw ID3
        if infogain > max_info_gain:
            max_info_gain = infogain
            max_info_gain_feature = feature

    #4 如果信息增益小于epsilon，置T为叶节点，以D中实例数最大的类Ck作为该节点的类标记，返回
    if max_info_gain < epsilon:
        [(most_large_class, _)] = Ck.most_common(1)
        return Node(col=None, results=most_large_class, branches=None)

    #5 对max_info_gain_feature每一个可能的值v，构建若干个非空子集， 将实例数最大的类作为标记构建子节点
    #6 对第i个子节点，以Di为训练集，features-max_info_gain_feature为特征集， 递归调用1-5

    features = features - set([max_info_gain_feature])
    feature_values = df[max_info_gain_feature].unique()
    branches = dict()
    for value in feature_values:
        subdf = df.query(max_info_gain_feature + "==@value")
        branches[value] = build_tree_ID3(subdf, features, label_col, epsilon)

    return Node(col=max_info_gain_feature,
                results= most_large_class,
                branches=branches)


def printTree(tree, depth=0):
    if tree is None: return
    if tree.branches is not None:
        for value, subtree in tree.branches.items():
            print("\t"*depth, "{0} = {1}".format(tree.col,value))
            printTree(subtree, depth+1 )
    else:
        print("\t"*depth, "decision is ", tree.results)




data = [
    dict(age='young', job='no', house='no', loan='normal', label='no'),
    dict(age='young', job='no', house='no', loan='good', label='no'),
    dict(age='young', job='yes', house='no', loan='good', label='yes'),
    dict(age='young', job='yes', house='yes', loan='normal', label='yes'),
    dict(age='young', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='good', label='no'),
    dict(age='mid', job='yes', house='yes', loan='good', label='yes'),
    dict(age='mid', job='no', house='yes', loan='verygood', label='yes'),
    dict(age='mid', job='no', house='yes', loan='verygood', label='yes'),
    dict(age='old', job='no', house='yes', loan='verygood', label='yes'),
    dict(age='old', job='no', house='yes', loan='good', label='yes'),
    dict(age='old', job='yes', house='no', loan='good', label='yes'),
    dict(age='old', job='yes', house='no', loan='verygood', label='yes'),
    dict(age='old', job='no', house='no', loan='normal', label='no'),

]


train=pd.DataFrame(data, columns=data[0].keys())
# root = build_tree_ID3(train, set(train.columns), 'label', 0)
root = build_tree_C45(train, set(train.columns), 'label', 0)

printTree(root)
# print("\n\n")

"""
 house = no
	 job = no
		 decision is  no
	 job = yes
		 decision is  yes
 house = yes
	 decision is  yes

    dict(age='young', job='no', house='no', loan='normal', label='no'),
    dict(age='young', job='no', house='no', loan='good', label='no'),
    dict(age='young', job='yes', house='no', loan='good', label='yes'),
    dict(age='young', job='yes', house='yes', loan='normal', label='yes'),
    dict(age='young', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='good', label='no'),
    dict(age='mid', job='yes', house='yes', loan='good', label='yes'),
    dict(age='mid', job='no', house='yes', loan='verygood', label='yes'),
    dict(age='mid', job='no', house='yes', loan='verygood', label='yes'),
    dict(age='old', job='no', house='yes', loan='verygood', label='yes'),
    dict(age='old', job='no', house='yes', loan='good', label='yes'),
    dict(age='old', job='yes', house='no', loan='good', label='yes'),
    dict(age='old', job='yes', house='no', loan='verygood', label='yes'),
    dict(age='old', job='no', house='no', loan='normal', label='no'),


信息增益做大的house, 含"no"的记录多，以"no"为下一个子集， 含"yes" 可以看到都label"yes"，所以简单成为叶节点

    dict(age='young', job='no', house='no', loan='normal', label='no'),
    dict(age='young', job='no', house='no', loan='good', label='no'),
    dict(age='young', job='yes', house='no', loan='good', label='yes'),
    dict(age='young', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='good', label='no'),
    dict(age='old', job='yes', house='no', loan='good', label='yes'),
    dict(age='old', job='yes', house='no', loan='verygood', label='yes'),
    dict(age='old', job='no', house='no', loan='normal', label='no'),

下一个信息增益最大的是job，含"no"大于含"yes"的，以"No"构建下一个子集， 含"yes"的叶碰巧都label"yes"，==》 成为叶节点

    dict(age='young', job='no', house='no', loan='good', label='no'),
    dict(age='young', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='normal', label='no'),
    dict(age='mid', job='no', house='no', loan='good', label='no'),
    dict(age='old', job='no', house='no', loan='normal', label='no'),
    
全部标记为"No"，结束



"""


"""
Some background of the data:
Imagine that this data is for a SaaS product we are selling. We offer users a
trail 14-days and at the end of the trial offer the users the ability to sign up
for a basic or premium offering.

1) Where the customer was referred from when they signed up for the trail (googole, slashdot, etc.) [Domain name string or (direct)]
2) Country of orgin (resolved by IP) [Country String]
3) Clicked on our FAQ link during the trail [Boolean]
4) How many application pages they viewed during the trial. [int]
5) What service they choose at the end of the trail. [None, Baisc, Premium string]

"""



my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['reddit','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['reddit','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['reddit','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]

my_data = pd.DataFrame(my_data, columns=['refer', 'country', 'FAQ', 'pages', 'label'])

# root = build_tree_ID3(my_data, set(my_data.columns), 'label', 0)
# root = build_tree_C45(my_data, set(my_data.columns), 'label', 0)
# printTree(root)


"""
 refer = slashdot
	 decision is  None
 refer = google
	 pages = 23
		 decision is  Premium
	 pages = 21
		 decision is  Premium
	 pages = 24
		 decision is  Premium
	 pages = 18
		 FAQ = no
			 decision is  None
		 FAQ = yes
			 decision is  Basic
 refer = reddit
	 FAQ = yes
		 decision is  Basic
	 FAQ = no
		 decision is  None
 refer = kiwitobes
	 country = France
		 decision is  Basic
	 country = UK
		 decision is  None
 refer = (direct)
	 country = New Zealand
		 decision is  None
	 country = UK
		 decision is  Basic
		 
"""