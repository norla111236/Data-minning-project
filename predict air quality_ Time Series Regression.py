# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:11:08 2020

@author: USER
"""
import sys  
import copy
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import cross_val_score
  # 讀檔
df = pd.read_csv('C:\\Users\\USER\\data minig\\hw3\\新竹_2019_1.csv',encoding = "big5")
  #刪除多於欄位
df = df.drop(index=[0])
df = df.drop(['測站                  '], axis=1)
 #去除包含特定異常值的
df.loc[df['0'].str.contains('[#|*|A|x|N]',regex=True),'0']=0
df.loc[df['1'].str.contains('[#|*|A|x|N]',regex=True),'1']=0
df.loc[df['2'].str.contains('[#|*|A|x|N]',regex=True),'2']=0
df.loc[df['3'].str.contains('[#|*|A|x|N]',regex=True),'3']=0
df.loc[df['4'].str.contains('[#|*|A|x|N]',regex=True),'4']=0
df.loc[df['5'].str.contains('[#|*|A|x|N]',regex=True),'5']=0
df.loc[df['6'].str.contains('[#|*|A|x|N]',regex=True),'6']=0
df.loc[df['7'].str.contains('[#|*|A|x|N]',regex=True),'7']=0
df.loc[df['8'].str.contains('[#|*|A|x|N]',regex=True),'8']=0
df.loc[df['9'].str.contains('[#|*|A|x|N]',regex=True),'9']=0
df.loc[df['10'].str.contains('[#|*|A|x|N]',regex=True),'10']=0
df.loc[df['11'].str.contains('[#|*|A|x|N]',regex=True),'11']=0
df.loc[df['12'].str.contains('[#|*|A|x|N]',regex=True),'12']=0
df.loc[df['13'].str.contains('[#|*|A|x|N]',regex=True),'13']=0
df.loc[df['14'].str.contains('[#|*|A|x|N]',regex=True),'14']=0
df.loc[df['15'].str.contains('[#|*|A|x|N]',regex=True),'15']=0
df.loc[df['16'].str.contains('[#|*|A|x|N]',regex=True),'16']=0
df.loc[df['17'].str.contains('[#|*|A|x|N]',regex=True),'17']=0
df.loc[df['18'].str.contains('[#|*|A|x|N]',regex=True),'18']=0
df.loc[df['19'].str.contains('[#|*|A|x|N]',regex=True),'19']=0
df.loc[df['20'].str.contains('[#|*|A|x|N]',regex=True),'20']=0
df.loc[df['21'].str.contains('[#|*|A|x|N]',regex=True),'21']=0
df.loc[df['22'].str.contains('[#|*|A|x|N]',regex=True),'22']=0
df.loc[df['23'].str.contains('[#|*|A|x|N]',regex=True),'23']=0

df['0'] = df['0'].astype(float)
df['1'] = df['1'].astype(float)
df['2'] = df['2'].astype(float)
df['3'] = df['3'].astype(float)
df['4'] = df['4'].astype(float)
df['5'] = df['5'].astype(float)
df['6'] = df['6'].astype(float)
df['7'] = df['7'].astype(float)
df['8'] = df['8'].astype(float)
df['9'] = df['9'].astype(float)
df['10'] = df['10'].astype(float)
df['11'] = df['11'].astype(float)
df['12'] = df['12'].astype(float)
df['13'] = df['13'].astype(float)
df['14'] = df['14'].astype(float)
df['15'] = df['15'].astype(float)
df['16'] = df['16'].astype(float)
df['17'] = df['17'].astype(float)
df['18'] = df['18'].astype(float)
df['19'] = df['19'].astype(float)
df['20'] = df['20'].astype(float)
df['21'] = df['21'].astype(float)
df['22'] = df['22'].astype(float)
df['23'] = df['23'].astype(float)

print(df[df==0].count())
  #測試用
# print(df.iat[0,2])
# print(df.iat[23,7])
# print(df.iat[23,8])
# print(df.iat[23,9])
# print(df.dtypes)
  #將零填入前後值的平均數
print(df.shape[0])
length = df.shape[0]
s = copy.deepcopy(df)
for j in range(0,length):
    for i in range(2,26):
            lower_num = 0
            upper_num = 0
            if(s.iat[j, i]==0):                
                # print(j)
                # print(i)
                upper = i+1
                lower = i-1
                if(upper>25):
                    upper = 2
                if(lower<2):
                    lower = 25
                if(s.iat[j, lower]!=0 and s.iat[j, upper]!=0):
                    lower_num = s.iat[j, lower]
                    upper_num = s.iat[j, upper]
#                     # df.loc[j][i] = (df.loc[j][lower] + df.loc[j][upper])/2
                elif(s.iat[j, lower]==0 and s.iat[j, upper]!=0):
                    upper_num = s.iat[j, upper]
                    for k in range(lower,1,-1):
                        if(s.iat[j, k]!=0):
                            lower_num = s.iat[j, k]
                            # print('lower1')
                            # print(s.iat[j, k])
                            break
                    if(lower==0):
                        for w in range(25,lower-1,-1):
                            if(s.iat[j, w]!=0):
                                lower_num = s.iat[j, w]
                                # print('lower2')
                                # print(s.iat[j, w])
                                break
                elif(s.iat[j, upper]==0 and s.iat[j, lower]!=0):
                    lower_num = s.iat[j, lower]
                    for k in range(upper,26):
                        if(s.iat[j, k]!=0):
                            upper_num = s.iat[j, k]
                            # print('upper1')
                            # print(s.iat[j, k])
                            break
                    if(upper==0):
                        for w in range(2,upper+1):
                            if(s.iat[j, w]!=0):
                                upper_num = s.iat[j, w]
                                # print('upper2')
                                # print(s.iat[j, w])
                                break
                elif(s.iat[j, upper]==0 and s.iat[j, lower]==0):
                    for k in range(upper,26):
                        if(s.iat[j, k]!=0):
                            upper_num = s.iat[j, k]
                            # print('upper1')
                            # print(s.iat[j, k])
                            break
                    if(upper==0):
                        for w in range(2,upper+1):
                            if(s.iat[j, w]!=0):
                                upper_num = s.iat[j, w]
                                # print('upper2')
                                # print(s.iat[j, w])
                                break
                    for k in range(lower,1,-1):
                        if(s.iat[j, k]!=0):
                            lower_num = s.iat[j, k]
                            # print('lower1')
                            # print(s.iat[j, k])
                            break
                    if(lower==0):
                        for w in range(25,lower-1,-1):
                            if(s.iat[j, w]!=0):
                                lower_num = s.iat[j, w]
                                # print('lower2')
                                # print(s.iat[j, w])
                                break
                # print("warning")
                # print((lower_num + upper_num)/2)
                # print(lower_num)
                # print(upper_num)
                test = (lower_num + upper_num)/2
                # print(test)
                # print(j)
                # print(i)
                # df.loc[j][i] = test
                df.iat[j, i] = test
    #             print(df.iat[j, i])
    # print("-----------------")

  #查看欄位，測試index
# print(type(df))
# print(df.index)
# print(type(df.index))
  #日期格式轉換
df['日期                  '] = pd.to_datetime(df['日期                  '])
  #日期時間選擇
df = df[(df['日期                  '] >=pd.to_datetime('2019/10/1 00:00')) & (df['日期                  '] <= pd.to_datetime('2019/12/31 00:00'))]
  #日期設為index
df = df.set_index('日期                  ') # 将date设置为index
# print(df)
  #有幾個為零
# print(df.apply(lambda x : x.value_counts().get(0,0),axis=1))
# print(df[df==0].count())
# print(df[df=='NR'].count())
  #零用NULL填補

df.fillna(value=0)
print(df[df==0].count())




