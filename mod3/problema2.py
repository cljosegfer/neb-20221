#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:29:40 2022

@author: jose
"""

import numpy as np
from anfis import anfis
import matplotlib.pyplot as plt

def f(x):
    return (1 + x[0]**(0.5) + x[1]**(-1) + x[2]**(-1.5))**2

# input
X_train = np.zeros(shape = (216, 3))
ind = 0
for i in range(1, 7):
    for j in range(1, 7):
        for k in range(1, 7):
            X_train[ind, 0] = i
            X_train[ind, 1] = j
            X_train[ind, 2] = k
            ind += 1
# shuffle
random = np.random.permutation(X_train.shape[0])
X = X_train[random]
y_train = np.array([f(x) for x in X_train]).reshape(-1, 1)
X_test = np.zeros(shape = (125, 3))
ind = 0
for i in range(1, 6):
    for j in range(1, 6):
        for k in range(1, 6):
            X_test[ind, 0] = i + 0.5
            X_test[ind, 1] = j + 0.5
            X_test[ind, 2] = k + 0.5
            ind += 1
y_test = np.array([f(x) for x in X_test]).reshape(-1, 1)

# anfis
n = 8
model = anfis(n = n, m = X_train.shape[1])
model.fit(X_train, y_train)

# report
yhat = model.predict(X_test).reshape(-1, 1)
mse = model.mse(X_test, y_test)
epm = (np.abs(y_test - yhat) / yhat).mean()
print('mse: {}, epm: {}'.format(mse, epm))

# plot
plt.plot(y_test)
plt.plot(yhat)

# log
plt.figure()
plt.plot(model.log)
