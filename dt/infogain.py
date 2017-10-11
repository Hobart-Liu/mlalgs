import pandas as pd
import numpy as np
import collections

data = [
    dict(age=1, job=0, house=0, loan=0, label=0),
    dict(age=1, job=0, house=0, loan=1, label=0),
    dict(age=1, job=1, house=0, loan=1, label=1),
    dict(age=1, job=1, house=1, loan=0, label=1),
    dict(age=1, job=0, house=0, loan=0, label=0),
    dict(age=2, job=0, house=0, loan=0, label=0),
    dict(age=2, job=0, house=0, loan=1, label=0),
    dict(age=2, job=1, house=1, loan=1, label=1),
    dict(age=2, job=0, house=1, loan=2, label=1),
    dict(age=2, job=0, house=1, loan=2, label=1),
    dict(age=3, job=0, house=1, loan=2, label=1),
    dict(age=3, job=0, house=1, loan=1, label=1),
    dict(age=3, job=1, house=0, loan=1, label=1),
    dict(age=3, job=1, house=0, loan=2, label=1),
    dict(age=3, job=0, house=0, loan=0, label=0),

]

train=pd.DataFrame(data, columns=data[0].keys())
print(train)

def calc_stat(df, label_name='label'):
    """
    提取统计信息

    :param df: DataFrame, 包含标签信息
    :param label_name: 标签所在的列名
    :return: label_tbls: dictionary, 相同分类值得坐标集合， 类值为KEY
             freq_tbls: dictionary, 每个feature不同取值 对应 标签不同取值的 个数统计, FEATURE值为Key
             value_per_feature: dictionary, 每个feature的可能取值， FEATURE值为Key

    """

    # 每个特征的可能取值
    values_per_feature = dict()
    for feature in df.columns:
        values_per_feature[feature] = df[feature].unique()

    # build frequency table
    # 每一个feature的取值对应分类的统计
    freq_tbls = dict()
    for feature in values_per_feature.keys():
        if feature == label_name: continue
        d = []
        for value in values_per_feature[feature]:
            d1 = []
            for label_value in values_per_feature[label_name]:
                d1.append(
                    len(
                        train.query(feature + ' == @value and ' + label_name + '==@label_value')
                    )
                )
            d.append(d1)
        freq_tbls[feature] = pd.DataFrame(d, columns=values_per_feature[label_name], index=values_per_feature[feature])

    # 分类的坐标信息
    label_tbls = collections.defaultdict(list)
    for i, c in enumerate(df[label_name]):
        label_tbls[c].append(i)

    return label_tbls, freq_tbls, values_per_feature


"""
test code 

label_tbls, freq_tbls , v_f= calc_stat(train)

print("Frequency Tables".center(20, "="))
for key, item in freq_tbls.items():
    print(key.center(20, '-'))
    print(item)
print("Label Tables".center(20, "="))
for key, item in label_tbls.items():
    print(key, item)
print("Values per Feature".center(20, "="))
for key, item in v_f.items():
    print(key, item)
    
    age  job  house  loan  label
0     1    0      0     0      0
1     1    0      0     1      0
2     1    1      0     1      1
3     1    1      1     0      1
4     1    0      0     0      0
5     2    0      0     0      0
6     2    0      0     1      0
7     2    1      1     1      1
8     2    0      1     2      1
9     2    0      1     2      1
10    3    0      1     2      1
11    3    0      1     1      1
12    3    1      0     1      1
13    3    1      0     2      1
14    3    0      0     0      0
==Frequency Tables==
--------age---------
   0  1
1  3  2
2  2  3
3  1  4
--------job---------
   0  1
0  6  4
1  0  5
-------house--------
   0  1
0  6  3
1  0  6
--------loan--------
   0  1
0  4  1
1  2  4
2  0  4
====Label Tables====
0 [0, 1, 4, 5, 6, 14]
1 [2, 3, 7, 8, 9, 10, 11, 12, 13]
=Values per Feature=
age [1 2 3]
job [0 1]
house [0 1]
loan [0 1 2]
label [0 1]

"""


def empirical_entropy(plist):
    """
    refer to page 62 李航 统计学习方法
    H(D) = -sum(Ck/D * log(Ck/D) / log(2)
    :param plist: Ck/D
    :return: empirical entropy scalar
    """
    # we first exclude 0, b/c definition 0*log0 == 0
    probs = [x for x in plist if x != 0]
    # we also check if there is negative value, since it is illeague
    tmp = [x for x in probs if x > 0 ]
    assert(len(tmp) == len(probs))

    return sum(
        [-p * np.log(p)/np.log(2) for p in probs]
    )

def empirical_conditional_entropy(freq_tbl):
    """
    Refer to page 62 李航 统计学习方法
    H(D|A) = - sum(Di/D * sum(Dik/Di * log(Dik/Di) / log(2) )
    :param freq_tbl: 一个特定feature的frequency table
    :return: H(D|A)
    """
    Di     = freq_tbl.sum(axis=1)
    Di_D   = Di / sum(Di)
    Dik_Di = freq_tbl.div(Di, axis=0)
    H_D_given_A = 0
    for i in range(len(freq_tbl)):
        tmp = [x * np.log(x) / np.log(2) for x in Dik_Di.iloc[i] if x != 0]
        H_D_given_A += Di_D.iloc[i] * sum(tmp)

    return -H_D_given_A

def info_gain(freq_tbl):
    D = freq_tbl.values.sum()
    Ck = freq_tbl.sum(axis=0)
    Ck_distribution = [x/D for x in Ck]
    H_D = empirical_entropy(Ck_distribution)
    H_D_given_A = empirical_conditional_entropy(freq_tbl)
    return H_D - H_D_given_A

def info_gain_ratio(freq_tbl):
    gain = info_gain(freq_tbl)
    D = freq_tbl.values.sum()
    Di = freq_tbl.sum(axis=1)
    H_a_D = -sum([x/D * np.log(x/D) / np.log(2) for x in Di])
    return gain/H_a_D

label_tbls, freq_tbls , v_f= calc_stat(train)

# calculate 数据D的经验熵H(D)
Ck = []
for key, item in label_tbls.items():
    Ck.append(len(item))
D = len(train)
probs = [x/D for x in Ck]
H_D = empirical_entropy(probs)
print("empirical entropy is", H_D)

for feature, item in freq_tbls.items():
    # D_features = list(item.sum(axis=1))
    # H_D_given_A = empirical_conditional_entropy(freq_tbls[feature])
    # print(feature, H_D, H_D_given_A, H_D-H_D_given_A)
    print(feature, "information gain", info_gain(freq_tbl=freq_tbls[feature]), info_gain_ratio(freq_tbls[feature]))
