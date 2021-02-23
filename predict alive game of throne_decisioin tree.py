# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:15:32 2020

@author: USER
"""
# 引入套件
import pandas as pd
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
# 讀檔
df = pd.read_csv('C:\\Users\\USER\\data minig\\hw1\\character-deaths.csv')
# print(df.isnull().sum())
# 缺值補零
df['Death Year'].fillna(value=0, inplace=True)
df['Book of Death'].fillna(value=0, inplace=True)
df['Death Chapter'].fillna(value=0, inplace=True)
# 既有數值轉1
df.loc[df['Death Year']>0,"Death Year"] = 1
df.loc[df['Book of Death']>0,"Book of Death"] = 1
df.loc[df['Death Chapter']>0,"Death Chapter"] = 1
# 將Allegiances轉成dummy特徵
pd.get_dummies(df['Allegiances'])
onehot_encoding = pd.get_dummies(df['Allegiances'], prefix = 'Allegiances')
df = df.drop('Allegiances', 1)
pd.concat([onehot_encoding, df],axis=1)
# 建立訓練集(75%)與測試集(25%) 
X = df.drop(labels=['Death Year','Book of Death','Death Chapter','Name','Book Intro Chapter'],axis=1)
y = df['Death Year']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# 建立決策樹並預測y_test
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=5, random_state=0)
clf.fit(X_train,y_train)
test_y_predicted = clf.predict(X_test)
print("預測結果：")
print(test_y_predicted)
print("標準答案：")
print(y_test)
#視覺化混淆矩陣
import seaborn as sns
C2 = confusion_matrix(y_test, test_y_predicted)
sns.heatmap(C2,annot=True)
#precision/recall計算
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
micro_precision = metrics.precision_score(y_test, test_y_predicted, average='micro')
macro_precision = metrics.precision_score(y_test, test_y_predicted, average='macro')
micro_recall = metrics.recall_score(y_test, test_y_predicted, average='micro')
macro_recall = metrics.recall_score(y_test, test_y_predicted, average='macro')
print("accuracy is :",accuracy)
print("micro precision is :",micro_precision)
print("macro precision is :",macro_precision)
print("micro recall is :",micro_recall)
print("macro recall is :",macro_recall)
#決策樹的圖
tree.plot_tree(clf) 
#輸出圖檔pdf檔
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("tree_graph")
















