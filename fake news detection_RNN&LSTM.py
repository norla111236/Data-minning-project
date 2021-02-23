# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:57:53 2020

@author: chein wen hui
"""

import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

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
nltk.download('stopwords') #下載停頓字
stop_words = set(stopwords.words('english')) #引入內建停頓字集
t = x_train.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])) #移除部分符號
tokenizer = Tokenizer() #將文本轉換為序列
tokenizer.fit_on_texts(t)
tokenizer.num_words=2000
max_words = tokenizer.num_words
s = tokenizer.word_index

#print(len(s))

for w in ['\'','im','2','u','am','3','4','!','5','1','x','n','w','us',',','.','?','-s','-ly','</s>','s']:
    stop_words.add(w) #對停頓字詞集做部分新增
#把此新增停頓字從train set 中移除
x_train = x_train.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])) 
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train,maxlen=24)
#把此新增停頓字從test set 中移除
x_test = x_test.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))   
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test,maxlen=24)

#%% 引入keras 的神經層套件
from keras.models import Sequential
from keras.layers.core import Dense,Dropout# ,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN,LSTM
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  #匯入layers模組
from keras.layers import ZeroPadding2D,Activation  #匯入layers模組

#%% RNN
modelRNN = Sequential()  #建立模型
modelRNN.name="RNN Model" #模型重新命名
#Embedding層將「數字list」轉換成「向量list」
modelRNN.add(Embedding(output_dim=32,   #輸出的維度是32，希望將數字list轉換為32維度的向量
                       input_dim=max_words+1,  #輸入的維度是max_words+1，也就是我們之前建立的字典是max_words+1字
                       input_length=24)) #數字list截長補短後都是18個數字
#加入Dropout，避免overfitting
modelRNN.add(Dropout(0.7)) 	#隨機在神經網路中放棄70%的神經元，避免overfitting
modelRNN.add(SimpleRNN(units=16)) #建立16個神經元的RNN層
modelRNN.add(Dense(units=256,activation='relu')) #建立256個神經元的隱藏層#ReLU激活函數
modelRNN.add(Dense(units=32,activation='relu')) #建立256個神經元的隱藏層#ReLU激活函數
modelRNN.add(Dense(units=1,activation='sigmoid'))#建立一個神經元的輸出層#Sigmoid激活函數
modelRNN.summary()
modelRNN.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

train_history = modelRNN.fit(x_train,y_train, 
         epochs=20, #執行10次訓練週期
         batch_size=150,#每一批次訓練100筆資料
         verbose=1,#verbose 顯示訓練過程
         validation_split=0.2)#validation_split =0.2 設定80%訓練資料、20%驗證資料

import matplotlib.pyplot as plt
print('mean loss:',np.mean(train_history.history["loss"]))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()

print('mean acc:',np.mean(train_history.history["accuracy"]))
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()
scores = modelRNN.evaluate(x_test, y_test,verbose=1)
scores[1]
RNNpredict = modelRNN.predict(x_test)
print("RNNpredict:")
print(RNNpredict)
print(" ")

#%% LSTM
modelLSTM = Sequential() #建立模型
modelLSTM.name="LSTM Model"
modelLSTM.add(Embedding(output_dim=32,
                       input_dim=max_words+1,  
                       input_length=24)) 
modelLSTM.add(Dropout(0.7))        
modelLSTM.add(LSTM(units=16)) 
modelLSTM.add(Dense(units=256,activation='relu')) 
modelLSTM.add(Dense(units=64,activation='relu')) 
modelLSTM.add(Dense(units=1,activation='sigmoid'))
modelLSTM .summary()
modelLSTM.compile(loss='binary_crossentropy',
     optimizer='adam',
     metrics=['accuracy']) 

train_history = modelLSTM.fit(x_train,y_train, 
         epochs=20, 
         batch_size=150,
         verbose=1,
         validation_split=0.2)

import matplotlib.pyplot as plt
print('mean loss:',np.mean(train_history.history["loss"]))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()

print('mean acc:',np.mean(train_history.history["accuracy"]))
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()

scores = modelLSTM .evaluate(x_test, y_test,verbose=1)
scores[1]
LSTMpredict = modelLSTM.predict(x_test)
print("LSTMpredict:")
print(LSTMpredict)
