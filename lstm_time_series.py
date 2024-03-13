# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 7)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import  mean_absolute_error

df = pd.read_csv('candy_production.csv')

df.head()

df.index = pd.to_datetime(df['observation_date'], dayfirst=True )
df['value'] = df['IPG3113N']
df = df.drop(['observation_date', 'IPG3113N'], axis=1)

df.head()

df.isnull().sum(axis=0)

sns.distplot(df['value'])

df.plot(figsize=(20, 8), fontsize=15)
plt.title('Сandy production from 1972-2017', fontsize=20)
plt.show()

df['year'] = df.index.year
df['month'] = df.index.month
df['weekday'] = df.index.weekday

df.head()

plt.figure(figsize=(15, 6))
sns.boxplot(x='year', y='value', data=df)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

dataset = df['value'].values

scaler = MinMaxScaler(feature_range = (0, 1))
dataset1 = scaler.fit_transform(dataset.reshape(-1, 1))
len(dataset1)

# Разделим выборку
train_size = int(len(dataset1) * 0.9)
test_size = len(dataset1) - train_size
train, test = dataset1[0:train_size], dataset1[train_size:len(dataset)]
print(len(train), len(test))

def create_dataset(dataset2, look_back):
  dataX=[]
  dataY =[]
  for i in range(len(dataset2)-look_back):
    a = dataset2[i:(i+look_back)]
    dataX.append(a)
    dataY.append(dataset2[i + look_back])
  return np.array(dataX), np.array(dataY)

look_back=3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=35, batch_size=1, verbose=2)

from sklearn.metrics import mean_squared_error
from math import *
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

plt.figure(figsize=(20, 8))
trainPredictPlot = np.empty_like(dataset1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+2*look_back:len(dataset1), :] = testPredict

plt.title('Production Forecast', fontsize=20)
plt.plot(scaler.inverse_transform(dataset1))
plt.plot(trainPredictPlot, color='green')
plt.plot(testPredictPlot,color='red')
plt.show()

df = pd.read_csv('candy_production.csv', index_col='observation_date', parse_dates=True)

# масштабировать данные в диапазоне от 0 до 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['IPG3113N'].values.reshape(-1,1))

# разделить данные на обучающий и тестовый наборы
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# функция для создания последовательностей модели LSTM
def create_sequences(data, look_back=12):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

# создавать последовательности для обучающего и тестового наборов
trainX, trainY = create_sequences(train_data)
testX, testY = create_sequences(test_data)

# измение формы входных данных, чтобы они были совместимы с моделью LSTM
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# создание LSTM-модели
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# подгонка модели по данным обучения
model.fit(trainX, trainY, epochs=50, batch_size=20)

# делать прогнозы на обучающем и тестовом множествах
train_predictions = scaler.inverse_transform(model.predict(trainX))
test_predictions = scaler.inverse_transform(model.predict(testX))

# создать долгосрочный прогноз на следующие 3 года
last_sequence = train_data[-12:]
long_term_forecast = []
current_month = df.index[-1]
for i in range(37):
    seq = last_sequence.reshape((1, 12, 1))
    prediction = model.predict(seq)
    long_term_forecast.append(prediction[0][0])
    last_sequence = np.append(last_sequence, prediction[0])[1:]
    current_month = current_month + pd.DateOffset(months=1)

# вернуть прогнозируемые значения к исходному масштабу
long_term_forecast = scaler.inverse_transform(np.array(long_term_forecast).reshape(-1,1))

plt.figure(figsize=(20, 8))
plt.title('Production 3 year Forecast', fontsize=20)
plt.plot(df.index, df['IPG3113N'], label='Известные данные / Known data')
plt.plot(pd.date_range(start=df.index[-1], periods=37, freq='M')[:len(long_term_forecast)], long_term_forecast, label='Долгосрочный прогноз на 3 года / Long-term forecast for 3 years')
plt.legend()
plt.show()
