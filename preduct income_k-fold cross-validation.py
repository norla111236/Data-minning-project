# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:15:32 2020

@author: USER
"""
# 引入套件
import pandas as pd
import seaborn as sns
import graphviz 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.metrics import average_precision_score
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import cross_val_score
# 讀檔
df = pd.read_csv('C:\\Users\\USER\\data minig\\hw2\\HW2data.csv')
# print(df.isnull().sum())

# 缺值補零
# df['Death Year'].fillna(value=0, inplace=True)
# # 既有數值轉1
# df.loc[df['Death Chapter']>0,"Death Chapter"] = 1

# # 轉成dummy特徵
pd.get_dummies(df['workclass'])
onehot_encoding = pd.get_dummies(df['workclass'], prefix = 'workclass')
df = df.drop('workclass', 1)
df = pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['education'])
onehot_encoding = pd.get_dummies(df['education'], prefix = 'education')
df = df.drop('education', 1)
pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['marital_status'])
onehot_encoding = pd.get_dummies(df['marital_status'], prefix = 'marital_status')
df = df.drop('marital_status', 1)
df = pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['occupation'])
onehot_encoding = pd.get_dummies(df['occupation'], prefix = 'occupation')
df = df.drop('occupation', 1)
pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['relationship'])
onehot_encoding = pd.get_dummies(df['relationship'], prefix = 'relationship')
df = df.drop('relationship', 1)
df = pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['race'])
onehot_encoding = pd.get_dummies(df['race'], prefix = 'race')
df = df.drop('race', 1)
pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['sex'])
onehot_encoding = pd.get_dummies(df['sex'], prefix = 'sex')
df = df.drop('sex', 1)
df = pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['native_country'])
onehot_encoding = pd.get_dummies(df['native_country'], prefix = 'native_country')
df = df.drop('native_country', 1)
df = pd.concat([onehot_encoding, df],axis=1)

pd.get_dummies(df['income'])
onehot_encoding = pd.get_dummies(df['income'], prefix = 'income')
df = df.drop('income', 1)
df = pd.concat([df, onehot_encoding],axis=1)

#check用
# print(onehot_encoding)
# print(df)
#check用預測去掉income, 決策留income_ <=50K，1表符合0表>50K
X = df.drop(labels=['income_ <=50K','income_ >50K'],axis=1)
y = df['income_ <=50K']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# #建立決策樹
clf = ensemble.RandomForestClassifier(n_estimators=100, criterion= 'gini', max_depth=None,  min_samples_split=2, min_samples_leaf=1);
forest_fit = clf.fit(X_train, y_train)
test_y_predicted = clf.predict(X_test)
#計算準確度
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
print("全部準確度為:")
print(accuracy)
C2 = confusion_matrix(y_test, test_y_predicted)
sns.heatmap(C2,annot=True)
# #套件下的準確度結果
# scores = cross_val_score(clf, X, y, cv=10)
# print("準確度為:")
# print(scores)
#k-fold 交叉驗證
scores = 0
def K_fold_CV(k, data):
    num_val_samples = len(data)
    scores = 0
    for i in range(k):
        print('Processing fold #', i)
        val_X = X[int(i*num_val_samples/k) : int((i+1)*num_val_samples/k)+1]
        val_y = y[int(i*num_val_samples/k) : int((i+1)*num_val_samples/k)+1]
        partial_X = pd.concat([X[: int(i*num_val_samples/k)+1], X[int((i+1)*num_val_samples/k)+1 :]], axis = 0)
        partial_y = pd.concat([y[: int(i*num_val_samples/k)+1], y[int((i+1)*num_val_samples/k)+1 :]], axis = 0)
        clf = ensemble.RandomForestClassifier(n_estimators=10);
        forest_fit = clf.fit(partial_X, partial_y)   
        partial_y_predicted = clf.predict(val_X) 
        accurac = metrics.accuracy_score(val_y, partial_y_predicted)
        print("第",i*10,"到", (i+1)*10,"筆資料作為測試集之準確度")
        print(accurac)
        scores += accurac   
    if i == 9:
        print("此",k,"筆資料平均為準確度為:",scores/10)         
        # df.drop(df.index, inplace=True) 清空dataframe
K_fold_CV(10, df)

#如何合併之測試
# df1 = df[4:7]
# df2 = df[32556:32561]
# df3 = pd.concat([df1, df2])
# print(df3)












#視覺化混淆矩陣
# C2 = confusion_matrix(y_test, test_y_predicted)
# sns.heatmap(C2,annot=True)
# #precision/recall計算
# accuracy = metrics.accuracy_score(y_test, test_y_predicted)
# micro_precision = metrics.precision_score(y_test, test_y_predicted, average='micro')
# macro_precision = metrics.precision_score(y_test, test_y_predicted, average='macro')
# micro_recall = metrics.recall_score(y_test, test_y_predicted, average='micro')
# macro_recall = metrics.recall_score(y_test, test_y_predicted, average='macro')
# print("accuracy is :",accuracy)
# print("micro precision is :",micro_precision)
# print("macro precision is :",macro_precision)
# print("micro recall is :",micro_recall)
# print("macro recall is :",macro_recall)
# #決策樹的圖
# tree.plot_tree(clf) 
# #輸出圖檔pdf檔
# dot_data = tree.export_graphviz(clf, out_file=None) 
# graph = graphviz.Source(dot_data) 
# graph.render("tree_graph")
















