import pandas as pd


# data = [
#     {"outlook": "sunny", "temp": "hot", "humidity": "high", "wind": "weak", "y": "no"},
#     {"outlook": "sunny", "temp": "hot", "humidity": "high", "wind": "strong", "y": "no"},
#     {"outlook": "overcast", "temp": "hot", "humidity": "high", "wind": "weak", "y": "yes"},
#     {"outlook": "rain", "temp": "mild", "humidity": "high", "wind": "weak", "y": "yes"},
#     {"outlook": "rain", "temp": "cool", "humidity": "normal", "wind": "weak", "y": "yes"},
#     {"outlook": "rain", "temp": "cool", "humidity": "normal", "wind": "strong", "y": "no"},
#     {"outlook": "overcast", "temp": "cool", "humidity": "normal", "wind": "strong", "y": "yes"},
#     {"outlook": "sunny", "temp": "mild", "humidity": "high", "wind": "weak", "y": "no"},
#     {"outlook": "sunny", "temp": "cool", "humidity": "normal", "wind": "weak", "y": "yes"},
#     {"outlook": "rain", "temp": "mild", "humidity": "normal", "wind": "weak", "y": "yes"},
#     {"outlook": "sunny", "temp": "mild", "humidity": "normal", "wind": "strong", "y": "yes"},
#     {"outlook": "overcast", "temp": "mild", "humidity": "high", "wind": "strong", "y": "yes"},
#     {"outlook": "overcast", "temp": "hot", "humidity": "normal", "wind": "weak", "y": "yes"},
#     {"outlook": "rain", "temp": "mild", "humidity": "high", "wind": "strong", "y": "no"}]

data = [
    {"X1": 1, "X2": 'S', 'y': -1},
    {"X1": 1, "X2": 'M', 'y': -1},
    {"X1": 1, "X2": 'M', 'y':  1},
    {"X1": 1, "X2": 'S', 'y':  1},
    {"X1": 1, "X2": 'S', 'y': -1},
    {"X1": 2, "X2": 'S', 'y': -1},
    {"X1": 2, "X2": 'M', 'y': -1},
    {"X1": 2, "X2": 'M', 'y':  1},
    {"X1": 2, "X2": 'L', 'y':  1},
    {"X1": 2, "X2": 'L', 'y':  1},
    {"X1": 3, "X2": 'L', 'y':  1},
    {"X1": 3, "X2": 'M', 'y':  1},
    {"X1": 3, "X2": 'M', 'y':  1},
    {"X1": 3, "X2": 'L', 'y':  1},
    {"X1": 3, "X2": 'L', 'y': -1},
]

train = pd.DataFrame(data)
print(train)



# 我们先统计一下有多少个feature， 以及每个feature可能的取值

f_v = dict()
for c in train.columns:
    f_v[c] = train[c].unique()

# build frequency table
# 每一个feature的取值对应分类的统计
freq_tbls = dict()
for k in f_v.keys():
    if k == 'y': continue
    d = []
    for v in f_v[k]:
        d1 = []
        for t in f_v['y']:

            d1.append(
                len(
                    train.query(k + ' == @v and y==@t')
                )
            )
        d.append(d1)
    freq_tbls[k] = pd.DataFrame(d, columns=f_v['y'], index=f_v[k])


print("Frequency Tables".center(20, "="))
for k in freq_tbls.keys():
    print(k.center(20, '-'))
    print(freq_tbls[k])


# fix Zero-Problem
# Laplace smoothing
# When an attribute value (Outlook=Overcast) doesn’t occur
# with every class value (Play Golf=no)
for k in freq_tbls.keys():
    freq_tbls[k] += 1

print("Frequency Tables".center(20, "="))
for k in freq_tbls.keys():
    print(k.center(20, '-'))
    print(freq_tbls[k])






# naive bayes 很直接
# test = {"outlook": "sunny", "temp": "cool", "humidity": "high", "wind": "strong"}
test = {"X1":2, "X2":'S'}

# 1) 计算先验概率P(Y=C_k)
pred = dict()
lamda = 1 # laplace smoothing for P(Y)
for v in f_v['y']:
    pred[v] = (sum(train['y'] == v)+ lamda) / (len(train) + lamda * len(f_v['y']))
print(pred)

# 2）计算条件概率 P(Xj=a_jl|Y=C_k)
# 3）对于给定的x， 计算 P(Y=C_k)*P(X1=cond_1|Y=C_k)*...*P(Xd=cond_d|Y=C_k)
# 4）确定x的类 y= argmax(P(C_k))
for key, item in test.items():
    for v in f_v['y']:
        # for each y, compute p(xn|y)
        p = freq_tbls[key].loc[item][v] / sum(freq_tbls[key][v])
        pred[v] = pred[v] * p

print(pred)

# Normalize
s = 0
for key, item in pred.items():
    s = item + s

for key, item in pred.items():
    pred[key] = item/s

print(pred)










