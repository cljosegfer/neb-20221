#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 23:06:46 2022

@author: jose
"""

import numpy as np
from anfis import anfis
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# input
# https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test
data = pd.read_csv('slump_test.data').values[:, 1:]

# train test
k = 10
size = len(data)
index = list(range(size))
np.random.shuffle(index)
step = round(size / k)
kfolds = [index[i:i+step] for i in range(0, size, step)]

k = 0
kfold = kfolds[k]
fold = np.ones(size, bool)
fold[kfold] = False

X = data[:, 0:-1]
y = data[:, -1]

X_test = data[np.invert(fold), 0:-1]
X_train = data[fold, 0:-1]
y_test = data[np.invert(fold), -1]
y_train = data[fold, -1]

# norm
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

# anfis
n = 32
model = anfis(n = n, m = X_train.shape[1])
model.fit(X_train, y_train, max_epochs = 20, alpha = 0.01)

# report
yhat = model.predict(X).reshape(-1, 1)
mse = model.mse(X_test, y_test)
epm = (np.abs(y - yhat) / yhat).mean()
print('mse: {}, epm: {}'.format(mse, epm))

# plot
plt.figure()
plt.plot(y)
plt.plot(yhat)

# log
plt.figure()
plt.plot(model.log)
