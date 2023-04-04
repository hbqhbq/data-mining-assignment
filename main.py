
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from urllib.parse import urlparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


from sklearn.feature_selection import VarianceThreshold


def url_to_domain(url):
    o = urlparse(url)
    domain = o.hostname
    return domain

dataset = pd.read_csv(r'C:\Users\hbq\Desktop\data mining\sports.csv', header=0, encoding='utf-8', dtype=str)

dataset = pd.DataFrame(dataset)

"""
df1 = dataset[385:635]
df2 = dataset[654:904]
train_data = pd.concat([df1, df2], join='inner',axis = 0)

print(train_data)
print(train_data.groupby('Label').count())
"""

list = []

for url in dataset['URL']:
    list.append(url_to_domain(url))

domain = pd.DataFrame(list, columns = ['site'])

dataset = dataset.merge(domain, how='right', left_index=True, right_index=True)
print(dataset)

dic = {'objective':'0', 'subjective':'1'}
dataset['Label'] = dataset['Label'].map(dic)

dataset = dataset.drop(['site','URL','TextID'],axis=1)



sel = VarianceThreshold(threshold = (.8 * (1- .8))).fit(dataset)
print(sel.get_support(indices = False))
drop_list = ['JJS','WRB', 'TOs','NNP', 'WP','ellipsis', 'sentencelast', 'sentence1st']
dataset.drop(drop_list, axis=1, inplace=True)


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

dataset1 = pd.DataFrame(dataset,dtype=np.int_)
corr_series = dataset1.corr(method='pearson').abs().unstack().sort_values(ascending=False)
print(corr_series[52:63])

dataset = dataset.drop(['VB','VBN','VBP','UH','FW'], axis=1)

print(dataset)


df1 = dataset[385:635]
df2 = dataset[654:904]
train_data = pd.concat([df1,df2], join='inner',axis = 0)


print(train_data.groupby('Label').count())

train_data = train_data.reset_index(drop = True)#重新设置索引



#print(X)
X_train = train_data.drop(['Label'], axis = 1)
y_train = train_data['Label']

X_test = dataset.drop(['Label'], axis = 1)
y_test = dataset['Label']



#filepath = r'C:\Users\hbq\Desktop\data mining\sports_train.csv'
#train_data.to_csv('C:\\Users\\hbq\\Desktop\\data mining\\sports_train.csv',index=False,header=True)  #index=False,header=False表示不保存行索引和列标题
#dataset.to_csv('C:\\Users\\hbq\\Desktop\\data mining\\sports_test.csv',index=False,header=True)

Cs = [0.1, 0.5, 1, 5, 10, 15, 20, 50]
cv = 5




"""
kf = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)
rf = RandomForestClassifier(random_state=13)
param_grid = {'n_estimators': [200, 500, 1000], 'max_depth': [4, 5, 6, 7, 8],
              'criterion': ['gini', 'entropy']}
clf = GridSearchCV(estimator=rf, cv=kf, param_grid=param_grid, scoring='roc_auc')
clf.fit(X_train, y_train)
print('Best params: {}'.format(clf.best_params_))
"""

rf = RandomForestClassifier(n_estimators=500, max_depth=8, criterion='gini')
rf.fit(X_train, y_train)
prediction_proba = rf.predict_proba(X_test)
print(prediction_proba)
prediction = rf.predict(X_test)
conf_mat = confusion_matrix(y_test, prediction)
acc = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction,average="binary", pos_label="1")
recall = recall_score(y_test, prediction,average="binary", pos_label="1")

print(conf_mat)
print('accuracy:',acc)
print('precison:', precision)
print('recall', recall)
print('roc_auc_score:',roc_auc_score(y_test, prediction_proba[:, 1]))

"""
dataset_1 = pd.DataFrame(dataset_1)
dataset_1 = dataset_1.drop(['URL','TextID'], axis=1)
print(dataset_1)

df1_1 = dataset_1[385:635]
df2_1 = dataset_1[654:904]
train_data_1 = pd.concat([df1_1,df2_1], join='inner',axis = 0)
train_data_1 = train_data_1.reset_index(drop = True)
print(train_data_1)


X_train_1 = train_data_1.drop(['Label'], axis = 1)
y_train_1 = train_data_1['Label']

X_test_1 = dataset_1.drop(['Label'], axis = 1)
y_test_1 = dataset_1['Label']

rf = RandomForestClassifier(n_estimators=500, max_depth=8, criterion='gini')
rf.fit(X_train_1, y_train_1)

prediction = rf.predict(X_test_1)
conf_mat = confusion_matrix(y_test_1, prediction)
acc = accuracy_score(y_test_1, prediction)

lrCV = LogisticRegressionCV(penalty='l1', Cs=Cs, cv=cv, solver='liblinear', max_iter=1000)
lrCV.fit(X_train_1, y_train_1)
score = lrCV.score(X_test_1, y_test_1)
prediction = lrCV.predict(X_test_1)

conf_mat = confusion_matrix(y_test_1, prediction)
acc = accuracy_score(y_test_1, prediction)
#scaler  =  preprocessing.StandardScaler().fit(X)


print(conf_mat)
print('accuracy:',acc)
"""