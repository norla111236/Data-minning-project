# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:57:53 2020

@author: chien wen hui
"""

import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.metrics import classification_report

#%% 資料前處理
test_data = pd.read_csv('C:\\Users\\USER\\data minig\\hw4\\test.csv', sep= "\t")
train_data = pd.read_csv('C:\\Users\\USER\\data minig\\hw4\\train.csv', sep= "\t")
train_data = train_data.drop([1615])
train_data.reset_index(drop=True, inplace=True)
label_data = pd.read_csv('C:\\Users\\USER\\data minig\\hw4\\sample_submission.csv', sep= "\t")
label_data[['id','label']]=label_data['id,label'].str.split(",",expand=True)

#測試用
print(test_data)
print(train_data)
print(label_data)

#分類測試與訓練集
x_train = train_data['text']
y_train = train_data['label']
x_test = test_data['text']
y_test = label_data['label']

#測試用
print(x_train)
print(y_train)
print(x_test)
print(y_test)

# 向量方法一
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)

# 向量方法二
tfidvectorizer = TfidfVectorizer(sublinear_tf=False,stop_words='english')
tfid_x_train = tfidvectorizer.fit_transform(x_train)
tfid_x_test = tfidvectorizer.transform(x_test)

#方法一二擇一套入即可

#%% xgboost
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train_tfidf, y_train)
xgboost_predict = xgbc.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, xgboost_predict)

print('<xgboost>')
print('xgboost_predict :')
print(xgboost_predict)
print("              Accuracy: %.2f" %accuracy)
print(classification_report(y_test, xgboost_predict))
print('')

#%% GBDT
from sklearn.ensemble import GradientBoostingClassifier

gbr = GradientBoostingClassifier(n_estimators=1000, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(X_train_tfidf, y_train)
gbr_predict = gbr.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, gbr_predict)

print('<GBDT>')
print('GBDT_predict :')
print(gbr_predict)
print("              Accuracy: %.2f" %accuracy)
print(classification_report(y_test, gbr_predict))
print('')

#%% LightGBM
import lightgbm as lgb

lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train_tfidf, y_train)
lgb_predict = lgbm.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, lgb_predict)

print('<LightGBM>')
print('LightGBM_predict :')
print(lgb_predict)
print("              Accuracy: %.2f" %accuracy)
print(classification_report(y_test, lgb_predict))
print('')

#%%






