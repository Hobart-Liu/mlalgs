import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# section 1: read data, shuffle, change label from string to float
filename = "sonar_all_data.csv"
colnames = ['c'+str(i) for i in range(60)]
colnames.append('type')
df = pd.read_csv(filename, index_col=None, header=None, names=colnames)
df = df.sample(frac=1).reset_index(drop=True)
df['lbl'] = 1.0
df.loc[df['type']=='R', 'lbl'] = 0.0
df.drop('type', axis=1, inplace=True)
df.astype(np.float32, inplace=True)
feature_names = ['c' + str(i) for i in range(60)]
label_name =['lbl']

# section 2: prep train and test data
test_x = df[:70][feature_names].get_values()
test_y = df[:70][label_name].get_values().ravel()
train_x = df[70:][feature_names].get_values()
train_y = df[70:][label_name].get_values().ravel()

# section 3: take a look at performance of sklearn decision tree and randomforest
clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)
print("Sklearn Decision Tree Classifier", clf.score(test_x, test_y))

rfclf = RandomForestClassifier(n_jobs=2)
rfclf.fit(train_x, train_y)
print("Sklearn Random Forest Classifier", rfclf.score(test_x, test_y))


# section 4: my first practice of random forest
m = 10
votes = [1/m] * m
num_train = len(train_x)
num_feat = len(train_x[0])


n = int(num_train * 0.6)
k = int(np.sqrt(num_feat))

index_of_train_data = np.arange(num_train)
index_of_train_feat = np.arange(num_feat)

clfs = [DecisionTreeClassifier() for _ in range(m)]
feats = []

for i, xclf in enumerate(clfs):
    np.random.shuffle(index_of_train_data)
    np.random.shuffle(index_of_train_feat)
    row_idx = index_of_train_data[:n]
    feat_idx = index_of_train_feat[:k]
    sub_train_x = train_x[row_idx,:][:, feat_idx]
    sub_train_y = train_y[row_idx]
    xclf.fit(sub_train_x, sub_train_y)
    feats.append(feat_idx)

pred = np.zeros(test_y.shape)

for xclf, feat, vote in zip(clfs, feats, votes):
    pred += xclf.predict(test_x[:, feat]) * vote

pred[pred  > 0.5] = 1.0
pred[pred <= 0.5] = 0.0
print("My home made Random Forest Classifier", sum(pred==test_y)/len(test_y))
